# cython: embedsignature=True
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# distutils: language = c++
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm

import scipy.integrate as integrate
from copy import copy
import numpy as np
# from math import comb
from scipy.special import comb
# cimport math
import itertools

cimport numpy as np
cimport cython

from cython.parallel import *
# cimport openmp

# include "pdf.pxi"
include 'integrate.pxi'



np.warnings.filterwarnings('ignore', '(overflow|invalid)')



def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz,
              double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1,
              double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim = 1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                        n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp == 1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier):
    return (p_outlier >= 0) & (p_outlier <= 1)


def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                double p_outlier=0, double w_outlier=0.1):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    for i in range(size):
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                     n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp

def wiener_like_rlddm(np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha, double v, 
                      double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):
            p = full_pdf(xs[i], ((qs[1] - qs[0]) * v), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf
            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


# JY added on 2021-12-05 for two-step tasks
def wiener_like_rlddm_2step(np.ndarray[double, ndim=1] x1, # 1st-stage RT
                      np.ndarray[double, ndim=1] x2, # 2nd-stage RT
                      np.ndarray[long,ndim=1] s1, # 1st-stage state
                      np.ndarray[long,ndim=1] s2, # 2nd-stage state
                      np.ndarray[long, ndim=1] response1,
                      np.ndarray[long, ndim=1] response2,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha,

                      double w, double gamma, double lambda_,


                      double v,
                      double sv, double a, double z, double sz, double t,
                      int nstates,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0,
                      ):
    cdef Py_ssize_t size = x1.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values


    cdef double dtQ1
    cdef double dtQ2

    cdef long s_
    cdef long a_

    cdef np.ndarray[double, ndim=1] x1s
    cdef np.ndarray[double, ndim=1] x2s
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses1
    cdef np.ndarray[long, ndim=1] responses2
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    cdef np.ndarray[long, ndim=1] s1s
    cdef np.ndarray[long, ndim=1] s2s

    # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
    # parameters added for two-step


    cdef np.ndarray[long, ndim=1] planets
    cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
    cdef np.ndarray[double, ndim=1] Qmb
    cdef double dtq
    cdef double rt
    cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
    cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses1 = response1[split_by == s]
        responses2 = response2[split_by == s]
        x1s = x1[split_by == s]
        x2s = x2[split_by == s]
        s1s = s1[split_by == s]
        s2s = s2[split_by == s]
        s_size = x1s.shape[0]
        qs_mf[:,0] = q
        qs_mf[:,1] = q

        qs_mb[:,0] = q
        qs_mb[:,1] = q

        # don't calculate pdf for first trial but still update q
        # if feedbacks[0] > qs[responses[0]]:
            # alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        # else:
            # alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # # received on current trial.
        # qs[responses[0]] = qs[responses[0]] + \
        #     alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(0, s_size):

            if counter[s1s[i]] > 0: # proceed with pdf only if the current 1st-stage state have been updated


                # 1st stage
                planets = state_combinations[s1s[i]]
                Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
                qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial

                dtq = qs[1] - qs[0]
                rt = x1s[i]
                if qs[0] > qs[1]:
                    dtq = -dtq
                    rt = -rt

                p = full_pdf(rt, (dtq * v), sv, a, z,
                             sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
                # If one probability = 0, the log sum will be -Inf
                p = p * (1 - p_outlier) + wp_outlier
                if p == 0:
                    return -np.inf
                sum_logp += log(p)


                # 2nd stage
                qs = qs_mb[s2s[i],:]
                dtq = qs[1] - qs[0]
                rt = x2s[i]
                if qs[0] > qs[1]:
                    dtq = -dtq
                    rt = -rt
                p = full_pdf(rt, (dtq * v), sv, a, z, sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)


                # 2022-01-04 meeting notes
                # keep threshold the same across stages
                # non decision time, drift rate might change between stages
                # Q) starting stage of the second stage might depend on the first stage drift rate? -> multistep DDM (Feng)
                # see whether non-decision time changes as a function of time or common/rare transition (use the condition separtion in HDDM)
                # Intertrial parameters -> run with and without, but if we're using the functional form then maybe not include



            # update Q values, regardless of pdf


            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            # if feedbacks[i] > qs[responses[i]]:
            #     alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            # else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            # qs[responses[i]] = qs[responses[i]] + \
            #     alfa * (feedbacks[i] - qs[responses[i]])


            dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
            qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

            dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2
            qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa * dtQ2 # delta update for qmb
            qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda_ * dtQ2 # eligibility trace


            # memory decay for unexperienced options in this trial

            for s_ in range(nstates):
                for a_ in range(2):
                    if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                        # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                        qs_mb[s_,a_] *= (1-gamma)

            for s_ in range(comb(nstates,2,exact=True)):
                for a_ in range(2):
                    if (s_ is not s1s[i]) or (a_ is not responses1[i]):
                        qs_mf[s_,a_] *= (1-gamma)

            counter[s1s[i]] += 1



    return sum_logp

# JY added on 2022-01-03 for simultaneous regression on two-step tasks
def wiener_like_rlddm_2step_reg(np.ndarray[double, ndim=1] x1, # 1st-stage RT
                      np.ndarray[double, ndim=1] x2, # 2nd-stage RT
                      # np.ndarray[long, ndim=1] isleft1, # whether left response 1st-stage,
                      # np.ndarray[long, ndim=1] isleft2, # whether left response 2nd-stage
                      np.ndarray[long,ndim=1] s1, # 1st-stage state
                      np.ndarray[long,ndim=1] s2, # 2nd-stage state
                      np.ndarray[long, ndim=1] response1,
                      np.ndarray[long, ndim=1] response2,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha,

                      # double w,
                      double gamma,
                      double lambda_,

                      double v0, double v1, double v2,
                      double v, # don't use second stage
                      # double sv,
                      double a,
                      double z0, double z1, double z2,
                      double z,
                      # double sz,
                      double t,
                      int nstates,
                      double v_qval, double z_qval,
                      double v_interaction, double z_interaction,
                      double two_stage,

                      double a_2,
                      double z_2,
                      double t_2,
                      double v_2,
                      double alpha2,
                      double w,

                      # double st,

                      double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0,
                      ):

    # if pos_alpha==100.00:
    #     pos_alfa = alpha
    # else:
    #     pos_alfa = pos_alpha

    if a==100.00: # if fixed threshold
        a = 1

    # if a_2 == 100.00: # if shared threshold
    #     a_2 = a
    # this part is in later if statement in 2nd stage


    # cdef double a = 1
    cdef double sz = 0
    cdef double st = 0
    cdef double sv = 0


    cdef Py_ssize_t size = x1.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef double alfa2

    cdef double gamma_
    cdef double lambda__

    # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
    cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values




    cdef double dtQ1
    cdef double dtQ2

    cdef double dtq_mb
    cdef double dtq_mf

    cdef long s_
    cdef long a_
    cdef double v_
    cdef double z_
    cdef double sig


    cdef np.ndarray[double, ndim=1] x1s
    cdef np.ndarray[double, ndim=1] x2s
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses1
    cdef np.ndarray[long, ndim=1] responses2
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    cdef np.ndarray[long, ndim=1] s1s
    cdef np.ndarray[long, ndim=1] s2s
    # cdef np.ndarray[long, ndim=1] isleft1s
    # cdef np.ndarray[long, ndim=1] isleft2s

    # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
    # parameters added for two-step


    cdef np.ndarray[long, ndim=1] planets
    cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
    cdef np.ndarray[double, ndim=1] Qmb
    cdef double dtq
    cdef double rt
    cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
    cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # if alpha2==100.00: # if either only 1st stage or don't share lr:


    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses1 = response1[split_by == s]
        responses2 = response2[split_by == s]
        x1s = x1[split_by == s]
        x2s = x2[split_by == s]
        s1s = s1[split_by == s]
        s2s = s2[split_by == s]

        # isleft1s = isleft1[split_by == s]
        # isleft2s = isleft2[split_by == s]

        s_size = x1s.shape[0]
        qs_mf[:,0] = q
        qs_mf[:,1] = q

        qs_mb[:,0] = q
        qs_mb[:,1] = q


        alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
        gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)
        if alpha2 != 100.00:
            alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
        else:
            alfa2 = alfa
        if lambda_ != 100.00:
            lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
        if w != 100.00:
            w = (2.718281828459**w) / (1 + 2.718281828459**w)

        # don't calculate pdf for first trial but still update q
        # if feedbacks[0] > qs[responses[0]]:
            # alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        # else:
            # alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # # received on current trial.
        # qs[responses[0]] = qs[responses[0]] + \
        #     alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(0, s_size):

            if counter[s1s[i]] > 0 and x1s[i]>0.15:
            # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)


                # 1st stage
                planets = state_combinations[s1s[i]]
                Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
                # qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial

                # if qval != 100: #


                # dtq = qs[1] - qs[0]
                dtq_mb = Qmb[0] - Qmb[1]
                dtq_mf = qs_mf[s1s[i],0] - qs_mf[s1s[i],1]
                if v == 100.00: # if v_reg
                    if v_qval == 0:

                        if v_interaction == 100.00: # if don't use interaction term
                            v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) # use both Qvals
                        else: # if use interaction term
                            v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) + (v_interaction * dtq_mb * dtq_mf)
                    elif v_qval == 1: # just mb
                        v_ = v0 + (dtq_mb * v1)
                    elif v_qval == 2:
                        v_ = v0 + (dtq_mf * v2) # just qmf
                else: # if don't use v_reg:
                    if v_qval == 0: # use both qmb and qmf
                        qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial
                        dtq = qs[1] - qs[0]
                        v_ = dtq * v
                    elif v_qval == 1:
                        v_ = dtq_mb * v
                    elif v_qval==2:
                        v_ = dtq_mf * v

                if z0 != 100.00: # if use z_reg:
                    if z_qval == 0:
                        if z_interaction == 100.00: # if don't use interaction term
                            z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) # use both Qvals
                        else:
                            z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) + (z_interaction * dtq_mb * dtq_mf)
                    elif z_qval == 1: # just mb
                        z_ = z0 + (dtq_mb * z1)
                    elif z_qval == 2:
                        z_ = z0 + (dtq_mf * z2) # just qmf
                    sig = 1/(1+np.exp(-z_))
                else: # if don't use z_reg:
                    sig = z

                rt = x1s[i]


                # if isleft1s[i] == 0: # if chosen right
                #     rt = -rt
                #     v_ = -v_

                # p = full_pdf(rt, (dtq * v), sv, a, z,
                #              sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
                p = full_pdf(rt, v_, sv, a, sig,
                             sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
                # If one probability = 0, the log sum will be -Inf
                p = p * (1 - p_outlier) + wp_outlier
                if p == 0:
                    return -np.inf
                sum_logp += log(p)


                # # # 2nd stage
                if two_stage == 1.00:

                    v_2_ = v if v_2==100.00 else v_2
                    a_2_ = a if a_2 == 100.00 else a_2
                    z_2_ = z if z_2 == 0.5 else z_2
                    t_2_ = t if t_2 == 100.00 else t_2

                    qs = qs_mb[s2s[i],:]
                    dtq = qs[1] - qs[0]
                    rt = x2s[i]
                    # if isleft2s[i] == 0:
                    #     dtq = -dtq
                    #     rt = -rt
                    p = full_pdf(rt, (dtq * v_2_), sv, a_2_, z_2_, sz, t_2_, st, err, n_st, n_sz, use_adaptive, simps_err)
                    # If one probability = 0, the log sum will be -Inf
                    p = p * (1 - p_outlier) + wp_outlier
                    if p == 0:
                        return -np.inf
                    sum_logp += log(p)


            # update Q values, regardless of pdf


            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            # if feedbacks[i] > qs[responses[i]]:
            #     alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            # else:


            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            # qs[responses[i]] = qs[responses[i]] + \
            #     alfa * (feedbacks[i] - qs[responses[i]])


            dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
            qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

            dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2
            qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
            if lambda_ != 100.00: # if using eligibility trace
                qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 # eligibility trace


            # memory decay for unexperienced options in this trial

            for s_ in range(nstates):
                for a_ in range(2):
                    if (s_ is not s2s[i]) or (a_ is not responses2[i]):
                        # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
                        qs_mb[s_,a_] *= (1-gamma_)

            for s_ in range(comb(nstates,2,exact=True)):
                for a_ in range(2):
                    if (s_ is not s1s[i]) or (a_ is not responses1[i]):
                        qs_mf[s_,a_] *= (1-gamma_)

            counter[s1s[i]] += 1



    return sum_logp


# # JY added on 2022-01-03 for simultaneous regression on two-step tasks
# def wiener_like_rlddm_2step_thinkact(np.ndarray[double, ndim=1] x1, # 1st-stage RT                      
#                       np.ndarray[double, ndim=1] x2, # 2nd-stage RT                     
#                       np.ndarray[long, ndim=1] isleft1, # whether left response 1st-stage, 
#                       np.ndarray[long, ndim=1] isleft2, # whether left response 2nd-stage  
#                       np.ndarray[long,ndim=1] s1, # 1st-stage state
#                       np.ndarray[long,ndim=1] s2, # 2nd-stage state
#                       np.ndarray[long, ndim=1] response1,
#                       np.ndarray[long, ndim=1] response2,
#                       np.ndarray[double, ndim=1] feedback,
#                       np.ndarray[long, ndim=1] split_by,
#                       double q, double alpha, double pos_alpha, 
#                       double gamma, 
#                       double lambda_, 

#                       double v0, double v1, double v2, 
#                       double v, # don't use second stage
#                       # double sv, 
#                       double a, 
#                       double z0, double z1, double z2,
#                       double z, 
#                       # double sz, 
#                       double t,
#                       int nstates,
#                       double qval,
#                       double two_stage,

#                       double a_2,
#                       double z_2, 
#                       double t_2,
#                       double v_2,
#                       double alpha2,
#                       double w,
#                       # double st, 

#                       double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
#                       double p_outlier=0, double w_outlier=0,
#                       ):

#     if a==100.00: # if fixed threshold
#         a = 1

#     cdef double sz = 0
#     cdef double st = 0
#     cdef double sv = 0

#     cdef Py_ssize_t size = x1.shape[0]
#     cdef Py_ssize_t i, j
#     cdef Py_ssize_t s_size
#     cdef int s
#     cdef double p
#     cdef double sum_logp = 0
#     cdef double wp_outlier = w_outlier * p_outlier
#     cdef double alfa
#     cdef double pos_alfa
#     cdef double alfa2

#     cdef double gamma_
#     cdef double lambda__

#     # cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
#     cdef np.ndarray[double, ndim=2] qs_mf = np.ones((comb(nstates,2,exact=True),2))*q # first-stage MF Q-values
#     cdef np.ndarray[double, ndim=2] qs_mb = np.ones((nstates, 2))*q # second-stage Q-values

#     cdef double dtQ1
#     cdef double dtQ2

#     cdef double dtq_mb
#     cdef double dtq_mf

#     cdef long s_
#     cdef long a_ 
#     cdef double v_
#     cdef double z_
#     cdef double sig

#     cdef np.ndarray[double, ndim=1] x1s
#     cdef np.ndarray[double, ndim=1] x2s
#     cdef np.ndarray[double, ndim=1] feedbacks
#     cdef np.ndarray[long, ndim=1] responses1
#     cdef np.ndarray[long, ndim=1] responses2
#     cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)    

#     cdef np.ndarray[long, ndim=1] s1s
#     cdef np.ndarray[long, ndim=1] s2s   
#     cdef np.ndarray[long, ndim=1] isleft1s
#     cdef np.ndarray[long, ndim=1] isleft2s        

#     # Added by Jungsun Yoo on 2021-11-27 for two-step tasks
#     # parameters added for two-step

#     cdef np.ndarray[long, ndim=1] planets
#     cdef np.ndarray[double, ndim=1] counter = np.zeros(comb(nstates,2,exact=True))
#     cdef np.ndarray[double, ndim=1] Qmb
#     cdef double dtq
#     cdef double rt
#     cdef np.ndarray[double, ndim=2] Tm = np.array([[0.7, 0.3], [0.3, 0.7]]) # transition matrix
#     cdef np.ndarray[long, ndim=2] state_combinations = np.array(list(itertools.combinations(np.arange(nstates),2)))

#     float delta_t = 0.001, # timesteps fraction of seconds
#     # cdef float thinking_t
#     cdef float t_particle
#     cdef float max_t    

#     t_particle = 0.0 # reset time
#     while t_particle <= max_t:




#         t_particle += delta_t


#     if not p_outlier_in_range(p_outlier):
#         return -np.inf

#     if pos_alpha==100.00:
#         pos_alfa = alpha
#     else:
#         pos_alfa = pos_alpha

#     # unique represent # of conditions
#     for j in range(unique.shape[0]):
#         s = unique[j]
#         # select trials for current condition, identified by the split_by-array
#         feedbacks = feedback[split_by == s]
#         responses1 = response1[split_by == s]
#         responses2 = response2[split_by == s]
#         x1s = x1[split_by == s]
#         x2s = x2[split_by == s]
#         s1s = s1[split_by == s]
#         s2s = s2[split_by == s]

#         isleft1s = isleft1[split_by == s]
#         isleft2s = isleft2[split_by == s]

#         s_size = x1s.shape[0]
#         qs_mf[:,0] = q
#         qs_mf[:,1] = q

#         qs_mb[:,0] = q
#         qs_mb[:,1] = q

#         alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
#         gamma_ = (2.718281828459**gamma) / (1 + 2.718281828459**gamma)
#         if alpha2 != 100.00:
#             alfa2 = (2.718281828459**alpha2) / (1 + 2.718281828459**alpha2)
#         else:
#             alfa2 = alfa            
#         if lambda_ != 100.00:
#             lambda__ = (2.718281828459**lambda_) / (1 + 2.718281828459**lambda_)
#         if w != 100.00:
#             w = (2.718281828459**w) / (1 + 2.718281828459**w)

#         # loop through all trials in current condition
#         for i in range(0, s_size):

#             if counter[s1s[i]] > 0 and x1s[i]>0.15: 
#             # proceed with pdf only if 1) the current 1st-stage state have been updated and 2) "plausible" RT (150 ms)

#                 for 


#                 # 1st stage
#                 planets = state_combinations[s1s[i]]
#                 Qmb = np.dot(Tm, [np.max(qs_mb[planets[0],:]), np.max(qs_mb[planets[1],:])])
#                 # qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial 

#                 # dtq = qs[1] - qs[0]
#                 dtq_mb = Qmb[0] - Qmb[1]
#                 dtq_mf = qs_mf[s1s[i],0] - qs_mf[s1s[i],1]
#                 if v == 100.00: # if v_reg
#                     if qval == 0:
#                         v_ = v0 + (dtq_mb * v1) + (dtq_mf * v2) # use both Qvals
#                     elif qval == 1: # just mb
#                         v_ = v0 + (dtq_mb * v1)
#                     elif qval == 2: 
#                         v_ = v0 + (dtq_mf * v2) # just qmf
#                 else: # if don't use v_reg:
#                     if qval == 0: # use both qmb and qmf
#                         qs = w * Qmb + (1-w) * qs_mf[s1s[i],:] # Update for 1st trial 
#                         dtq = qs[1] - qs[0]
#                         v_ = dtq * v
#                     elif qval == 1:
#                         v_ = dtq_mb * v
#                     elif qval==2:
#                         v_ = dtq_mf * v 

#                 if z0 != 100.00: # if use z_reg:
#                     if qval == 0:
#                         z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2) # use both Qvals
#                     elif qval == 1: # just mb
#                         z_ = z0 + (dtq_mb * z1)
#                     elif qval == 2: 
#                         z_ = z0 + (dtq_mf * z2) # just qmf
#                     sig = 1/(1+np.exp(-z_))
#                 else: # if don't use z_reg:
#                     sig = z


#                 # z_ = z0 + (dtq_mb * z1) + (dtq_mf * z2)
#                 # sig =  np.where(z_<0, np.exp(z_)/(1+np.exp(z_)), 1/(1+np.exp(-z_))) # perform sigmoid on z to bound it [0,1]
#                 # sig = 1/(1+np.exp(-z_))
                
#                 rt = x1s[i]
#                 # if qs[0] > qs[1]:
#                 #     dtq = -dtq
#                 #     rt = -rt

#                 if isleft1s[i] == 0: # if chosen right
#                     rt = -rt
#                     v_ = -v_

#                 # p = full_pdf(rt, (dtq * v), sv, a, z,
#                 #              sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
#                 p = full_pdf(rt, v_, sv, a, sig,
#                              sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)                
#                 # If one probability = 0, the log sum will be -Inf
#                 p = p * (1 - p_outlier) + wp_outlier
#                 if p == 0:
#                     return -np.inf
#                 sum_logp += log(p)


#                 # # # 2nd stage
#                 if two_stage == 1.00:

#                     v_2_ = v if v_2==100.00 else v_2
#                     a_2_ = a if a_2 == 100.00 else a_2 
#                     z_2_ = z if z_2 == 0.5 else z_2 
#                     t_2_ = t if t_2 == 100.00 else t_2                                        

#                     qs = qs_mb[s2s[i],:]
#                     dtq = qs[1] - qs[0]
#                     rt = x2s[i]
#                     if isleft2s[i] == 0:
#                     # if qs[0] > qs[1]:
#                         dtq = -dtq
#                         rt = -rt           
#                     p = full_pdf(rt, (dtq * v_2_), sv, a_2_, z_2_, sz, t_2_, st, err, n_st, n_sz, use_adaptive, simps_err)
#                     # If one probability = 0, the log sum will be -Inf
#                     p = p * (1 - p_outlier) + wp_outlier
#                     if p == 0:
#                         return -np.inf
#                     sum_logp += log(p)


#             # update Q values, regardless of pdf    


#             # get learning rate for current trial. if pos_alpha is not in
#             # include it will be same as alpha so can still use this
#             # calculation:
#             # if feedbacks[i] > qs[responses[i]]:
#             #     alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
#             # else:


#             # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
#             # received on current trial.
#             # qs[responses[i]] = qs[responses[i]] + \
#             #     alfa * (feedbacks[i] - qs[responses[i]])


#             dtQ1 = qs_mb[s2s[i],responses2[i]] - qs_mf[s1s[i], responses1[i]] # delta stage 1
#             qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + alfa * dtQ1 # delta update for qmf

#             dtQ2 = feedbacks[i] - qs_mb[s2s[i],responses2[i]] # delta stage 2 
#             qs_mb[s2s[i], responses2[i]] = qs_mb[s2s[i],responses2[i]] + alfa2 * dtQ2 # delta update for qmb
#             if lambda_ != 100.00: # if using eligibility trace
#                 qs_mf[s1s[i], responses1[i]] = qs_mf[s1s[i], responses1[i]] + lambda__ * dtQ2 # eligibility trace        


#             # memory decay for unexperienced options in this trial

#             for s_ in range(nstates):
#                 for a_ in range(2):
#                     if (s_ is not s2s[i]) or (a_ is not responses2[i]):
#                         # qs_mb[s_, a_] = qs_mb[s_, a_] * (1-gamma)
#                         qs_mb[s_,a_] *= (1-gamma_)

#             for s_ in range(comb(nstates,2,exact=True)):
#                 for a_ in range(2):
#                     if (s_ is not s1s[i]) or (a_ is not responses1[i]):
#                         qs_mf[s_,a_] *= (1-gamma_)
           
#             counter[s1s[i]] += 1



#     return sum_logp



def wiener_like_rl(np.ndarray[long, ndim=1] response,
                   np.ndarray[double, ndim=1] feedback,
                   np.ndarray[long, ndim=1] split_by,
                   double q, double alpha, double pos_alpha, double v, double z,
                   double err=1e-4, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                   double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = response.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double drift
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha
        
    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        s_size = responses.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):

            drift = (qs[1] - qs[0]) * v

            if drift == 0:
                p = 0.5
            else:
                if responses[i] == 1:
                    p = (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)
                else:
                    p = 1 - (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)

            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf

            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


def wiener_like_multi(np.ndarray[double, ndim=1] x, v, sv, a, z, sz, t, st, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]
            if abs(x[i]) != 999.:
                p = full_pdf(x[i], params_iter['v'],
                             params_iter['sv'], params_iter['a'], params_iter['z'],
                             params_iter['sz'], params_iter['t'], params_iter['st'],
                             err, n_st, n_sz, use_adaptive, simps_err)
                p = p * (1 - p_outlier) + wp_outlier
            elif x[i] == 999.:
                p = prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])
            else: # x[i] == -999.
                p = 1 - prob_ub(params_iter['v'], params_iter['a'], params_iter['z'])

            sum_logp += log(p)

        return sum_logp


def wiener_like_multi_rlddm(np.ndarray[double, ndim=1] x, 
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, v, sv, a, z, sz, t, st, alpha, double err, multi=None,
                      int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t ij
    cdef Py_ssize_t s_size
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef int s
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st, 'alpha':alpha}
        params_iter = copy(params)
        qs[0] = q
        qs[1] = q
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]

            if (i != 0):
                if (split_by[i] != split_by[i-1]):
                    qs[0] = q
                    qs[1] = q

            p = full_pdf(x[i], params_iter['v'] * (qs[1] - qs[0]),
                         params_iter['sv'], params_iter['a'], params_iter['z'],
                         params_iter['sz'], params_iter[
                             't'], params_iter['st'],
                         err, n_st, n_sz, use_adaptive, simps_err)
            p = p * (1 - p_outlier) + wp_outlier
            sum_logp += log(p)

            alfa = (2.718281828459**params_iter['alpha']) / (1 + 2.718281828459**params_iter['alpha'])   
            qs[response[i]] = qs[response[i]] + alfa * (feedback[i] - qs[response[i]])

        return sum_logp


def gen_rts_from_cdf(double v, double sv, double a, double z, double sz, double t,
                     double st, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


# JY added for simulation with factorial design

        # rts = hddm.wfpt.gen_rts_from_cdf_factorial(            
        #     params["v"],
        #     params["sv"],
        #     params["a"],
        #     params["z"],
        #     params["sz"],
        #     params["t"],
        #     params["st"],
        #     # JY added on 2022-02-17 for factorial
        #     params["a2"], 
        #     params["t2"], 
        #     params["z0"], 
        #     params["z1"], 
        #     params["z2"], 
        #     params["v0"], 
        #     params["v1"], 
        #     params["v2"],
        #     # ===========
        #     size,
        #     range_[0],
        #     range_[1],
        #     dt,
        #     )

def gen_rts_from_cdf_factorial(double v, double sv, double a, double z, double sz, double t,
                     double st, 
                     double a2, double t2,                      
                     double z0, double z1, double z2, 
                     double v0, double v1, double v2, 
                     int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v,
                            double sv, double a, double z, double sz, double t, double st, double t_min,
                            double t_max, double err, int n_st=10, int n_sz=10, bint use_adaptive=1,
                            double simps_err=1e-8):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(size, nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                         n_st, n_sz, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf

    # add the log likelihood of the contaminations
    sum_logp += n_cont * log(0.5 * 1. / (t_max - t_min))

    return sum_logp

def gen_cdf_using_pdf(double v, double sv, double a, double z, double sz, double t, double st, double err,
                      int N=500, double time=5., int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    """
    generate cdf vector using the pdf
    """
    if (sv < 0) or (a <= 0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z + sz / 2. > 1) or \
            (z - sz / 2. < 0) or (t - st / 2. < 0) or (t < 0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError(
            "at least one of the parameters is out of the support")

    cdef np.ndarray[double, ndim = 1] x = np.linspace(-time, time, 2 * N + 1)
    cdef np.ndarray[double, ndim = 1] cdf_array = np.empty(x.shape[0], dtype=np.double)
    cdef int idx

    # compute pdf on the real line
    cdf_array = pdf_array(x, v, sv, a, z, sz, t, st, err, 0,
                          n_st, n_sz, use_adaptive, simps_err, p_outlier, w_outlier)

    # integrate
    cdf_array[1:] = integrate.cumtrapz(cdf_array)

    # normalize
    cdf_array /= cdf_array[x.shape[0] - 1]

    return x, cdf_array


def split_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] data):

    # get length of data
    cdef int N = (len(data) - 1) / 2

    # lower bound is reversed
    cdef np.ndarray[double, ndim = 1] x_lb = -x[:N][::-1]
    cdef np.ndarray[double, ndim = 1] lb = data[:N][::-1]
    # lower bound is cumulative in the wrong direction
    lb = np.cumsum(np.concatenate([np.array([0]), -np.diff(lb)]))

    cdef np.ndarray[double, ndim = 1] x_ub = x[N + 1:]
    cdef np.ndarray[double, ndim = 1] ub = data[N + 1:]
    # ub does not start at 0
    ub -= ub[0]

    return (x_lb, lb, x_ub, ub)


# NEW WITH NN-EXTENSION

#############
# Basic MLP Likelihoods
def wiener_like_nn_mlp(np.ndarray[float, ndim = 1] rt,
                       np.ndarray[float, ndim = 1] response,
                       np.ndarray[float, ndim = 1] params,
                       double p_outlier = 0,
                       double w_outlier = 0,
                       network = None):

    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]
    cdef float log_p = 0
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

def wiener_like_nn_mlp_pdf(np.ndarray[float, ndim = 1] rt,
                            np.ndarray[float, ndim = 1] response,
                            np.ndarray[float, ndim = 1] params,
                            double p_outlier = 0, 
                            double w_outlier = 0,
                            bint logp = 0,
                            network = None):
    
    cdef Py_ssize_t size = rt.shape[0]
    cdef Py_ssize_t n_params = params.shape[0]

    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
    cdef float ll_min = -16.11809

    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile(params, (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([rt, response], axis = 1)

    # Call to network:
    if p_outlier == 0: # ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else: # ddm_model
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    if logp == 0:
        log_p = np.exp(log_p) # shouldn't be called log_p anymore but no need for an extra array here
    return log_p


################
# Regression style likelihoods: (Can prob simplify and make all mlp likelihoods of this form)

def wiener_like_multi_nn_mlp(np.ndarray[float, ndim = 2] data,
                             double p_outlier = 0, 
                             double w_outlier = 0,
                             network = None):
                             #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.sum(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p 

def wiener_like_multi_nn_mlp_pdf(np.ndarray[float, ndim = 2] data,
                                 double p_outlier = 0, 
                                 double w_outlier = 0,
                                 network = None):
                                 #**kwargs):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0: # previous ddm_model
        log_p = np.squeeze(np.core.umath.maximum(network.predict_on_batch(data), ll_min))
    else:
        log_p = np.squeeze(np.log(np.exp(np.core.umath.maximum(network.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p

###########
# Basic CNN likelihoods

#def wiener_like_cnn_2(np.ndarray[long, ndim = 1] x, 
#                      np.ndarray[long, ndim = 1] response, 
#                      np.ndarray[float, ndim = 1] parameters,
#                      double p_outlier = 0, 
#                      double w_outlier = 0,
#                      **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i 
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#
#    # Call to network:
#    return log_p
#
#def wiener_pdf_cnn_2(np.ndarray[long, ndim = 1] x, 
#                     np.ndarray[long, ndim = 1] response, 
#                     np.ndarray[float, ndim = 1] parameters,
#                     double p_outlier = 0, 
#                     double w_outlier = 0,
#                     bint logp = 0,
#                     **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef np.ndarray[float, ndim = 1] log_p = np.zeros(size, dtype = np.float32)
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #print(pred.shape)
#    #log_p = 0
#    for i in range(size):
#        if response[i] == 0:
#            log_p[i] += np.log(pred[0, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p[i] += np.log(pred[0, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    if logp == 0:
#        log_p = np.exp(log_p)
#
#    # Call to network:
#    return log_p
#
#def wiener_like_reg_cnn_2(np.ndarray[long, ndim = 1] x, 
#                          np.ndarray[long, ndim = 1] response, 
#                          np.ndarray[float, ndim = 2] parameters,
#                          double p_outlier = 0, 
#                          double w_outlier = 0,
#                          bint logp = 0,
#                          **kwargs):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef float log_p = 0
#    cdef np.ndarray[float, ndim = 2] pred = kwargs['network'](parameters)
#    #log_p = 0
#    #print(pred.shape)
#    #print(pred)
#    for i in range(size):
#        if response[i] == 0:
#            log_p += np.log(pred[i, 2 * x[i]] * (1 - p_outlier) + w_outlier * p_outlier)
#        else: 
#            log_p += np.log(pred[i, 2 * x[i] + 1] * (1 - p_outlier) + w_outlier * p_outlier)
#    
#    # Call to network:
#    return log_p
#
#