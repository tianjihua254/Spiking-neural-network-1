# -*- coding:utf-8 -*-
def euqations(use_weight_dependence, post_pre):
    """  STDP synaptic traces，STDP随时间变换的公式, 可以把pre、post看作是突触的变量，随时间变换
         determine which STDP rule to use，确定使用哪种STDP规则"""

    eqs_stdp_ee = '''
                w:siemens
                dpre/dt = -pre / tc_pre_ee : siemens (event-driven)
                dpost/dt = -post / tc_post_ee : siemens (event-driven)
                '''
    # setting STDP update rule 四种改进的STDP更新规则
    if use_weight_dependence:
        if post_pre:
            eqs_stdp_pre_ee = '''
            ge+=w
            pre = 1.*nS
            w = clip(w - nu_ee_pre * post * w ** exp_ee_pre, 0, wmax_ee)
            '''
            eqs_stdp_post_ee = '''
            w = clip(w + nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post, 0, wmax_ee)
            post = 1.*nS
            '''
        else:
            eqs_stdp_pre_ee = '''
            ge+=w
            pre = 1.*nS
            '''
            eqs_stdp_post_ee = '''
            w = clip(w + nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post, 0, wmax_ee)
            post = 1.*nS
            '''
    else:  # no weight dependence
        if post_pre:
            eqs_stdp_pre_ee = '''
            ge+=w
            pre = 1.*nS
            w = clip(w - nu_ee_pre * post, 0, wmax_ee)
            '''
            eqs_stdp_post_ee = '''
            w = clip(w + nu_ee_post * pre, 0, wmax_ee)
            post = 1.*nS
            '''
        else:
            eqs_stdp_pre_ee = '''
            ge+=w
            pre = 1.*nS
            '''
            eqs_stdp_post_ee = '''
            w = clip(w + nu_ee_post * pre, 0, wmax_ee)
            post = 1.*nS
            '''
    return (eqs_stdp_ee, eqs_stdp_pre_ee, eqs_stdp_post_ee)
