"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlddm, wiener_like_rlddm_2step_reg # wiener_like_rlddm_2step,
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
        self.v_reg = kwargs.pop("v_reg", True) # added for regression in two-step task
        self.z_reg = kwargs.pop("z_reg", False)
        self.a_fix = kwargs.pop("a_fix", True)

        self.two_stage = kwargs.pop("two_stage", False) # whether to RLDDM just 1st stage or both stages
        self.sep_alpha = kwargs.pop("sep_alpha", False) # use different learning rates for second stage

        self.v_sep_q = kwargs.pop("v_sep_q", False) # In 1st stage, whether to use Qmf/Qmb separately for v (drift rate) regression   
        self.v_qmb = kwargs.pop("v_qmb", False) # Given sep_q, True = qmb, False = Qmf
        self.v_interaction = kwargs.pop("v_interaction", False) # whether to include interaction term for v
        

        self.z_sep_q = kwargs.pop("z_sep_q", False) # In 1st stage, whether to use Qmf/Qmb separately for z (starting point) regression    
        self.z_qmb = kwargs.pop("z_qmb", False) # Given sep_q, True = qmb, False = Qmf
        self.z_interaction = kwargs.pop("z_interaction", False) # whether to include interaction term for z


        self.a_share = kwargs.pop("a_share", False) # whether to share a btw 1st & 2nd stage (if a!=1)
        self.v_share = kwargs.pop("v_share", False) # whether to share v btw 1st & 2nd stage (if v!=reg)
        self.z_share = kwargs.pop("z_share", False) # whether to share z btw 1st & 2nd stage (if z!=reg)
        self.t_share = kwargs.pop("t_share", False) # whether to share t btw 1st & 2nd stage


        # JY added on 2022-03-15 for configuring starting point bias
        # if second-stage starting point depends on 1st-stage parameters

        self.z_2_depend = kwargs.pop("z_2_depend", False)  # whether z_2 depends on previous stage



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

            if (not self.v_reg) and (not self.v_sep_q):
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
            if self.two_stage and self.sep_alpha:
                knodes.update(
                    self._create_family_normal_non_centered(
                        "alpha2",
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
                if self.v_sep_q:
                    if self.v_qmb: # == 'mb': # just use MB Qvalues
                        knodes.update(
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
                    else:
                        knodes.update(
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
                else: # if both 
                    if self.v_interaction: # if include interaction term for v1 and v2
                        knodes.update(
                            self._create_family_normal_non_centered(
                                "v_interaction",
                                value=0,
                                g_mu=0.2,
                                g_tau=3 ** -2,
                                std_lower=1e-10,
                                std_upper=10,
                                std_value=0.1,
                        )
                    )     


                    knodes.update(
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
                    knodes.update(
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
                self._create_family_invlogit(
                    "z0", value=0.5, g_tau=0.5 ** -2, std_std=0.05)                    

                )
                if self.z_sep_q:
                    if self.z_qmb: # == 'mb': # just use MB Qvalues                
                        knodes.update(
                        self._create_family_invlogit(
                            "z1", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
                        )
                    else:
                        knodes.update(
                        self._create_family_invlogit(
                            "z2", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
                        )
                else: # if both
                    if self.z_interaction:
                        knodes.update(
                        self._create_family_invlogit(
                            "z_interaction", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  


                        ) 
                    knodes.update(
                    self._create_family_invlogit(
                        "z1", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
                    )                    
                    knodes.update(
                    self._create_family_invlogit(
                        "z2", value=0.5, g_tau=0.5 ** -2, std_std=0.05)  
                    )
            if self.z_2_depend: # if second-stage starting point depends on first stage v -> only z std is needed, use z0 as std?
                knodes.update(
                self._create_family_invlogit(
                    "z_sigma", value=0.5, g_tau=0.5 ** -2, std_std=0.05)

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
            if self.two_stage and self.sep_alpha:
                knodes.update(
                    self._create_family_normal(
                        "alpha2",
                        value=0,
                        g_mu=0.2,
                        g_tau=3 ** -2,
                        std_lower=1e-10,
                        std_upper=10,
                        std_value=0.1,
                    )
                )
            if (not self.v_reg) and (not self.v_sep_q):
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
                if self.v_sep_q:
                    if self.v_qmb: # == 'mb': # just use MB Qvalues
                        knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                            )
                        )
                    else: 
                        knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "v2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                            )
                        )
                else: # if both

                    if self.v_interaction:
                        knodes.update(
                            self._create_family_normal_normal_hnormal(
                                "v_interaction", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                                )
                            ) 


                    knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "v1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )
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
                if self.z_sep_q:
                    if self.z_qmb:
                        knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                            )
                        )
                    else:
                        knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                            )
                        )
                else: # if both
                    if self.z_interaction: 
                        knodes.update(
                        self._create_family_normal_normal_hnormal(
                            "z_interaction", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                            )
                        )

                    knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "z1", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                        )
                    )

                    knodes.update(
                    self._create_family_normal_normal_hnormal(
                        "z2", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                    )
                    )
            if self.z_2_depend: # if second-stage starting point depends on first stage v -> only z std is needed, use z0 as std?
                knodes.update(
                self._create_family_normal_normal_hnormal(
                    "z_sigma", value=0, g_tau=50 ** -2, std_std=10 # uninformative prior
                    )
                )

            # if self.z1:

            # if self.z2:


        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents["alpha"] = knodes["alpha_bottom"]
        wfpt_parents["pos_alpha"] = knodes["pos_alpha_bottom"] if self.dual else 100.00

        wfpt_parents["alpha2"] = knodes["alpha2_bottom"] if self.two_stage and self.sep_alpha else 100.00

        if (not self.v_reg) and (not self.v_sep_q):
            wfpt_parents["w"] = knodes["w_bottom"]
        else:
            wfpt_parents["w"] = 100.00

        wfpt_parents["gamma"] = knodes["gamma_bottom"]
        wfpt_parents["lambda_"] = knodes["lambda__bottom"] if self.lambda_ else 100.00

        # wfpt_parents["v0"] = knodes["v0_bottom"] if self.v_reg else 100.00
        # wfpt_parents["v1"] = knodes["v1_bottom"] if self.v_reg else 100.00
        # wfpt_parents["v2"] = knodes["v2_bottom"] if self.v_reg else 100.00

        # wfpt_parents["z0"] = knodes["z0_bottom"] if self.z_reg else 100.00
        # wfpt_parents["z1"] = knodes["z1_bottom"] if self.z_reg else 100.00
        # wfpt_parents["z2"] = knodes["z2_bottom"] if self.z_reg else 100.00



        if self.v_reg: # if using v_regression 
            wfpt_parents['v'] = 100.00
            wfpt_parents["v0"] = knodes["v0_bottom"]
            if self.v_sep_q:
                if self.v_qmb: # == 'mb': # just use MB Qvalues
                    wfpt_parents['v_qval'] = 1.00
                    wfpt_parents["v1"] = knodes["v1_bottom"]
                    wfpt_parents["v2"] = 100.00
                    wfpt_parents['v_interaction'] = 100.00
                    
                # elif self.qval == 'mf':
                else:
                    wfpt_parents['v_qval'] = 2.00
                    wfpt_parents["v1"] = 100.00
                    wfpt_parents["v2"] = knodes["v2_bottom"]
                    wfpt_parents['v_interaction'] = 100.00
                    

            else:

                if self.v_interaction:
                    wfpt_parents['v_interaction'] = knodes['v_interaction_bottom']
                else:
                    wfpt_parents['v_interaction'] = 100.00

                wfpt_parents['v_qval'] = 0.00
                wfpt_parents["v1"] = knodes["v1_bottom"]
                wfpt_parents["v2"] = knodes["v2_bottom"]
        
        else: # if not using v_regression: just multiplying v * Q
            wfpt_parents["v0"] = 100.00    
            wfpt_parents["v1"] = 100.00
            wfpt_parents["v2"] = 100.00
            wfpt_parents["v_interaction"] = 100.00

            if self.v_sep_q:
                if self.v_qmb: # == 'mb': # just use MB Qvalues
                    wfpt_parents['v_qval'] = 1.00
                # elif self.qval == 'mf':
                else:
                    wfpt_parents['v_qval'] = 2.00

            else:
                wfpt_parents['v_qval'] = 0.00           

        if self.z_reg:
            wfpt_parents["z0"] = knodes["z0_bottom"]
            if self.z_sep_q:
                if self.z_qmb:
                    wfpt_parents['z_qval'] = 1.00
                    wfpt_parents["z1"] = knodes["z1_bottom"]
                    wfpt_parents["z2"] = 100.00
                    wfpt_parents['z_interaction'] = 100.00
                else:
                    wfpt_parents['z_qval'] = 2.00
                    wfpt_parents["z1"] = 100.00
                    wfpt_parents["z2"] = knodes["z2_bottom"]
                    wfpt_parents['z_interaction'] = 100.00
            else:
                if self.z_interaction: 
                    wfpt_parents['z_interaction'] = knodes['z_interaction_bottom']
                else:
                    wfpt_parents['z_interaction'] = 100.00
                wfpt_parents['z_qval'] = 0.00
                wfpt_parents["z1"] = knodes["z1_bottom"]
                wfpt_parents["z2"] = knodes["z2_bottom"]
        else: # if not using z_regression
            wfpt_parents["z0"] = 100.00
            wfpt_parents["z1"] = 100.00
            wfpt_parents["z2"] = 100.00
            wfpt_parents['z_interaction'] = 100.00
            if self.z_sep_q:
                if self.z_qmb: # == 'mb': # just use MB Qvalues
                    wfpt_parents['z_qval'] = 1.00
                # elif self.qval == 'mf':
                else:
                    wfpt_parents['z_qval'] = 2.00

            else:
                wfpt_parents['z_qval'] = 0.00              

        # if self.z_2_depend:
        # wfpt_parents['z_sigma'] = knodes['z_sigma'] if self.z_2_depend else 100.00
        # wfpt_parents['z_sigma'] = 100.00

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
            # elif not self.a_share and self.a_fix:
            #     wfpt_parents['a_2'] = 100.00
            # if self.z_share:
            #     wfpt_parents['z_2'] = 100.00
            if self.t_share:
                wfpt_parents['t_2'] = 100.00
        else:
            wfpt_parents['two_stage'] = 0.00
            # since only first-stage is modeled, none of v_2,a_2,t_2,z_2 is used
            wfpt_parents['v_2'] = 100.00
            wfpt_parents['a_2'] = 100.00
            # wfpt_parents['z_2'] = 100.00
            wfpt_parents['t_2'] = 100.00



        # wfpt_parents["z"] = knodes["z_bottom"] if "z" in self.include else 0.5

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(
            self.wfpt_rl_class,
            "wfpt",
            observed=True,
            col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", ],
            # col_name=["split_by", "feedback", "response1", "response2", "rt1", "rt2",  "q_init", "state1", "state2", "isleft1", "isleft2"],
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

# def wienerRL_like_2step(x, v, alpha, pos_alpha, w, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0):
#
#     wiener_params = {
#         "err": 1e-4,
#         "n_st": 2,
#         "n_sz": 2,
#         "use_adaptive": 1,
#         "simps_err": 1e-3,
#         "w_outlier": 0.1,
#     }
#     wp = wiener_params
#     response1 = x["response1"].values.astype(int)
#     response2 = x["response2"].values.astype(int)
#     state1 = x["state1"].values.astype(int)
#     state2 = x["state2"].values.astype(int)
#
#     # isleft1 = x["isleft1"].values.astype(int)
#     # isleft2 = x["isleft2"].values.astype(int)
#
#
#     q = x["q_init"].iloc[0]
#     feedback = x["feedback"].values.astype(float)
#     split_by = x["split_by"].values.astype(int)
#
#
#     # YJS added for two-step tasks on 2021-12-05
#     # nstates = x["nstates"].values.astype(int)
#     nstates = max(x["state2"].values.astype(int)) + 1
#
#
#     return wiener_like_rlddm_2step(
#         x["rt1"].values,
#         x["rt2"].values,
#         state1,
#         state2,
#         response1,
#         response2,
#         feedback,
#         split_by,
#         q,
#         alpha,
#         pos_alpha,
#         w, # added for two-step task
#         gamma, # added for two-step task
#         lambda_, # added for two-step task
#
#         v,
#         sv,
#         a,
#         z,
#         sz,
#         t,
#         nstates,
#         st,
#         p_outlier=p_outlier,
#         **wp
#     )
# def wienerRL_like_2step_reg(x, v, alpha, pos_alpha, w, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0):
# def wienerRL_like_2step_reg(x, v, v0, v1, v2, alpha, pos_alpha, gamma, lambda_, sv, a, z, sz, t, st, p_outlier=0): # regression ver1: without bounds
# def wienerRL_like_2step_reg(x, v0, v1, v2, alpha, pos_alpha, gamma, lambda_, z0, z1, z2,t, p_outlier=0): # regression ver2: bounded, a fixed to 1
def wienerRL_like_2step_reg(x, v0, v1, v2, v_interaction, z0, z1, z2, z_interaction, lambda_, alpha, pos_alpha, gamma, a,z,t,v, a_2, z_2, t_2,v_2,alpha2, v_qval,z_qval,two_stage, w, z_sigma, p_outlier=0): # regression ver2: bounded, a fixed to 1

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

    # isleft1 = x["isleft1"].values.astype(int)
    # isleft2 = x["isleft2"].values.astype(int)


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

        # isleft1,
        # isleft2,

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
        v_qval,
        z_qval,
        v_interaction, 
        z_interaction, 
        two_stage,

        a_2,
        z_2,
        t_2,
        v_2,
        alpha2,
        w,
        z_sigma,
        # st,
        p_outlier=p_outlier,
        **wp
    )
# WienerRL = stochastic_from_dist("wienerRL", wienerRL_like)
# WienerRL = stochastic_from_dist("wienerRL_2step", wienerRL_like_2step)
WienerRL = stochastic_from_dist("wienerRL_2step_reg", wienerRL_like_2step_reg)

