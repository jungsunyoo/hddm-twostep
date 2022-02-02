"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlddm, wiener_like_rlddm_2step, wiener_like_rlddm_2step_reg
from collections import OrderedDict


class HDDMrl(HDDM):
    """HDDM model that can be used for two-armed bandit tasks."""

    # just 2-stage rlddm

    # def __init__(self, *args, **kwargs):
    #     self.non_centered = kwargs.pop("non_centered", False)
    #     self.dual = kwargs.pop("dual", False)
    #     self.alpha = kwargs.pop("alpha", True)
    #     self.w = kwargs.pop("w", True) # added for two-step task
    #     self.gamma = kwargs.pop("gamma", True) # added for two-step task
    #     self.lambda_ = kwargs.pop("lambda_", True) # added for two-step task
    #     self.wfpt_rl_class = WienerRL

    #     super(HDDMrl, self).__init__(*args, **kwargs)
    #

    # 2-stage rlddm regression

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop("non_centered", False)
        self.dual = kwargs.pop("dual", False)
        self.alpha = kwargs.pop("alpha", True)
        self.gamma = kwargs.pop("gamma", True) # added for two-step task

        self.lambda_ = kwargs.pop("lambda_", False) # added for two-step task
        self.v_reg = kwargs.pop("v_reg", False) # added for regression in two-step task
        self.z_reg = kwargs.pop("z_reg", False)
        self.a_fix = kwargs.pop("a_fix", False)

        self.two_stage = kwargs.pop("two_stage", False) # whether to RLDDM just 1st stage or both stages
        self.sep_q = kwargs.pop("sep_q", False) # In 1st stage, whether to use Qmf/Qmb separately    
        # self.sep_reg_qmf = kwargs.pop("sep_reg_qmf", False) # In 1st stage, whether to just model MF-Qval
        self.qval = kwargs.pop("qval", False) # specify which Qval to use (inputs = "mb", "mf")

        self.a_share = kwargs.pop("a_share", False) # whether to share a btw 1st & 2nd stage (if a!=1)
        self.v_share = kwargs.pop("v_share", False) # whether to share v btw 1st & 2nd stage (if v!=reg)
        self z_share = kwargs.pop("z_share", False) # whether to share z btw 1st & 2nd stage (if z!=reg)
        self.t_share = kwargs.pop("t_share", False) # whether to share t btw 1st & 2nd stage

        self.wfpt_rl_class = WienerRL

        super(HDDMrl, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        # params = ["t"]
        # if "p_outlier" in self.include:
        #     params.append("p_outlier")
        # # if "z" in self.include:
        # if not self.v_reg:
        #     params.append("v")
        # if not self.z_reg:
        #     params.append("z")
        # if not self.a_fix:
        #     params.append("a")

        # include = set(params)

        knodes = super(HDDMrl, self)._create_stochastic_knodes(include)
        if self.non_centered:
            print("setting learning rate parameter(s) to be non-centered")
            if self.alpha:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if (not self.v_reg) and (not self.sep_q):    
                knodes.update(
                    self._create_family_normal_non_centered(
                        "w",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if self.dual:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "pos_alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )
            
            if self.gamma:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "gamma",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                ) 
            if self.lambda_:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "lambda_",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if self.v_reg:
                knodes.update(
                    # self._create_family_normal_non_centered(
                    #     "v0",
                    #     value=0,
                    #     g_tau=50 ** -2, 
                    #     # std_std=10,
                    #     # g_mu=0.2,
                    #     # g_tau=3 ** -2,
                    #     std_lower=1e-10,
                    #     std_upper=10,
                    #     std_value=0.1,
                    # )
                    self._create_family_normal_non_centered(
                        "v0",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )                    
                )

            # if self.v1:
                knodes.update(
                    # self._create_family_normal_non_centered(
                    #     "v1",
                    #     value=0,
                    #     g_tau=50 ** -2, 
                    #     # std_std=10,
                    #     # g_mu=0.2,
                    #     # g_tau=3 ** -2,
                    #     std_lower=1e-10,
                    #     std_upper=10,
                    #     std_value=0.1,
                    # )
                    self._create_family_normal_non_centered(
                        "v1",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )                    
                )
            # if self.v2:
                knodes.update(
                    # self._create_family_normal_non_centered(
                    #     "v2",
                    #     value=0,
                    #     g_tau=50 ** -2, 
                    #     # std_std=10,
                    #     # g_mu=0.2,
                    #     # g_tau=3 ** -2,
                    #     std_lower=1e-10,
                    #     std_upper=10,
                    #     std_value=0.1,
                    # )
                    self._create_family_normal_non_centered(
                        "v2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )                   
                )

            if self.z_reg:
                knodes.update(
                    # self._create_family_normal_non_centered(
                    #     "v0",
                    #     value=0,
                    #     g_tau=50 ** -2, 
                    #     # std_std=10,
                    #     # g_mu=0.2,
                    #     # g_tau=3 ** -2,
                    #     std_lower=1e-10,
                    #     std_upper=10,
                    #     std_value=0.1,
                    # )
                self._create_family_invlogit(
                    "z0", value=0.5, g_tau=0.5 ** -2, std_std=0.05)                    

                )

            # if self.z1:
                knodes.update(
                self._create_family_invlogit(
                    "z1", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
            )
            # if self.z2:
                knodes.update(
                self._create_family_invlogit(
                    "z2", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
            )


        else:
            if self.alpha:
                knodes.update(
                    self._create_family_normal(
                        "alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if (not self.v_reg) and (not self.sep_q):
                knodes.update(
                    self._create_family_normal(
                        "w",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )


            if self.dual:
                knodes.update(
                    self._create_family_normal(
                        "pos_alpha",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if self.gamma:
                knodes.update(
                    self._create_family_normal(
                        "gamma",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )
            if self.lambda_:
                knodes.update(
                    self._create_family_normal(
                        "lambda_",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )

            if self.v_reg:

                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "v0", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )

            # if self.v1:
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "v1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )
            # if self.v2:
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "v2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )
            # )

            if self.z_reg:
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "z0", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )

            # if self.z1:
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "z1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )
            # if self.z2:
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "z2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                )
            )

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents["alpha"] = knodes["alpha_bottom"]
        wfpt_parents["pos_alpha"] = knodes["pos_alpha_bottom"] if self.dual else 100.00


        if (not self.v_reg) and (not self.sep_q):    
            wfpt_parents["w"] = knodes["w_bottom"]
        else:
            wfpt_parents["w"] = 100.00

        wfpt_parents["gamma"] = knodes["gamma_bottom"]
        wfpt_parents["lambda_"] = knodes["lambda__bottom"] if self.lambda_ else 100.00

        wfpt_parents["v0"] = knodes["v0_bottom"] if self.v_reg else 100.00
        wfpt_parents["v1"] = knodes["v1_bottom"] if self.v_reg else 100.00
        wfpt_parents["v2"] = knodes["v2_bottom"] if self.v_reg else 100.00

        wfpt_parents["z0"] = knodes["z0_bottom"] if self.z_reg else 100.00
        wfpt_parents["z1"] = knodes["z1_bottom"] if self.z_reg else 100.00
        wfpt_parents["z2"] = knodes["z2_bottom"] if self.z_reg else 100.00

        if self.sep_q:
            if self.qval == 'mb': # just use MB Qvalues
                wfpt_parents['qval'] = 1.00
            elif self.qval == 'mf':
                wfpt_parents['qval'] = 2.00
        else:
            wfpt_parents['qval'] = 0.00

        if self.v_reg:
            wfpt_parents['v'] = 100.00
            if self.qval == 'mb': # just use Qmb
                wfpt_parents["v2"] = 100.00
            elif self.qval == 'mf': # just use Qmf
                wfpt_parents["v1"] = 100.00


        # if self.z_reg:
        #     wfpt_parents['z'] = 100.00
        if self.a_fix:
            wfpt_parents['a'] = 100.00   

        if self.two_stage: # two stage RLDDM
            wfpt_parents['two_stage'] = 1.00
            if self.v_share:
                wfpt_parents['v_2'] = 100.00
            if self.a_share:
                wfpt_parents['a_2'] = 100.00
            if self.z_share:
                wfpt_parents['z_2'] = 100.00
            if self.t_share:
                wfpt_parents['t_2'] = 100.00
        else:
            wfpt_parents['two_stage'] = 0.00



        # wfpt_parents["z"] = knodes["z_bottom"] if "z" in self.include else 0.5

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.wfpt_rl_class,
            "wfpt",
            observed=True,
            # col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", ],
            col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", "isleft1", "isleft2"],
            **wfpt_parents
        )


def wienerRL_like(x, v, alpha, pos_alpha, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    wp = wiener_params
    response = x["response"].values.astype(int)
    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)
    return wiener_like_rlddm(
        x["rt"].values,
        response,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha,
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        p_outlier=p_outlier,
        **wp
    )

def wienerRL_like_2step(x, v, alpha, pos_alpha, w, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    wp = wiener_params
    response1 = x["response1"].values.astype(int)
    response2 = x["response2"].values.astype(int)
    state1 = x["state1"].values.astype(int)
    state2 = x["state2"].values.astype(int)

    isleft1 = x["isleft1"].values.astype(int)
    isleft2 = x["isleft2"].values.astype(int)


    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)


    # YJS added for two-step tasks on 2021-12-05
    # nstates = x["nstates"].values.astype(int)
    nstates = max(x["state2"].values.astype(int)) + 1


    return wiener_like_rlddm_2step(
        x["rt1"].values,
        x["rt2"].values,
        state1,
        state2,
        response1,
        response2,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha, 
        w, # added for two-step task
        gamma, # added for two-step task 
        lambda_, # added for two-step task 

        v,
        sv,
        a,
        z,
        sz,
        t,
        nstates,
        st,
        p_outlier=p_outlier,
        **wp
    )
# def wienerRL_like_2step_reg(x, v, alpha, pos_alpha, w, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0):
# def wienerRL_like_2step_reg(x, v, v0, v1, v2, alpha, pos_alpha, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0): # regression ver1: without bounds
# def wienerRL_like_2step_reg(x, v0, v1, v2, alpha, pos_alpha, gamma, lambda_, z0, z1, z2,t, p_outlier=0): # regression ver2: bounded, a fixed to 1
def wienerRL_like_2step_reg(x, v0, v1, v2, z0, z1, z2,lambda_, alpha, pos_alpha, gamma, a,z,t,v, a_2, z_2, t_2,v_2,alpha2, qval,two_stage, w, p_outlier=0): # regression ver2: bounded, a fixed to 1

    wiener_params = {
        "err": 1e-4,
        "n_st": 2,
        "n_sz": 2,
        "use_adaptive": 1,
        "simps_err": 1e-3,
        "w_outlier": 0.1,
    }
    wp = wiener_params
    response1 = x["response1"].values.astype(int)
    response2 = x["response2"].values.astype(int)
    state1 = x["state1"].values.astype(int)
    state2 = x["state2"].values.astype(int)

    isleft1 = x["isleft1"].values.astype(int)
    isleft2 = x["isleft2"].values.astype(int)


    q = x["q_init"].iloc[0]
    feedback = x["feedback"].values.astype(float)
    split_by = x["split_by"].values.astype(int)


    # JY added for two-step tasks on 2021-12-05
    # nstates = x["nstates"].values.astype(int)
    nstates = max(x["state2"].values.astype(int)) + 1

    # # JY added for which q-value to use (if sep, qmb or qmf)
    # qval = 0 # default: simultaneous

    # if


    return wiener_like_rlddm_2step_reg(
        x["rt1"].values,
        x["rt2"].values,

        isleft1,
        isleft2,

        state1,
        state2,
        response1,
        response2,
        feedback,
        split_by,
        q,
        alpha,
        pos_alpha, 
        # w, # added for two-step task
        gamma, # added for two-step task 
        lambda_, # added for two-step task 
        v0, # intercept for first stage rt regression
        v1, # slope for mb
        v2, # slobe for mf
        v, # don't use second stage for now
        # sv,
        a,
        z0, # bias: added for intercept regression 1st stage
        z1, # bias: added for slope regression mb 1st stage
        z2, # bias: added for slope regression mf 1st stage
        z,
        # sz,
        t,
        nstates,
        qval,
        two_stage,

        a_2,
        z_2,
        t_2,
        v_2,
        alpha2,
        w, 
        # st,
        p_outlier=p_outlier,
        **wp
    )
# WienerRL = stochastic_from_dist("wienerRL", wienerRL_like)
# WienerRL = stochastic_from_dist("wienerRL_2step", wienerRL_like_2step)
WienerRL = stochastic_from_dist("wienerRL_2step_reg", wienerRL_like_2step_reg)

