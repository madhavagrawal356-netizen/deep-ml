import math
from statistics import NormalDist


def analyze_ab_test(control_outcomes, treatment_outcomes,
                    confidence_level=0.95, min_detectable_effect=0.02):
    if not control_outcomes or not treatment_outcomes:
        return {}

    nd = NormalDist()
    nc, nt = len(control_outcomes), len(treatment_outcomes)
    sc, st = sum(control_outcomes), sum(treatment_outcomes)
    pc, pt = sc / nc, st / nt
    delta = pt - pc

    # relative lift
    if pc == 0:
        rel_lift = 'inf' if pt > 0 else 0.0
    else:
        rel_lift = round(100 * delta / pc, 2)

    # pooled two-proportion z-test
    p_bar = (sc + st) / (nc + nt)
    var = p_bar * (1 - p_bar) * (1 / nc + 1 / nt)
    denom = math.sqrt(var)
    z = delta / denom if denom > 0 else 0.0
    p_value = 2 * (1 - nd.cdf(abs(z)))

    # confidence interval (unpooled SE)
    alpha = 1 - confidence_level
    z_crit = nd.inv_cdf(1 - alpha / 2)
    se = math.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    ci = (round(delta - z_crit * se, 4), round(delta + z_crit * se, 4))

    stat_sig = p_value < alpha
    prac_sig = abs(delta) >= min_detectable_effect

    # required sample size (80% power)
    q_bar = (pc + pt) / 2
    if min_detectable_effect <= 0 or q_bar in (0, 1):
        required_n = 0
    else:
        required_n = math.ceil(
            2 * ((z_crit + 0.84) / min_detectable_effect) ** 2 * q_bar * (1 - q_bar)
        )

    # recommendation
    if stat_sig and prac_sig and delta > 0:
        rec = 'launch_treatment'
    elif stat_sig and (delta < 0 or not prac_sig):
        rec = 'keep_control'
    else:
        rec = 'continue_testing'

    return {
        'control_rate': round(pc, 4),
        'treatment_rate': round(pt, 4),
        'control_size': nc,
        'treatment_size': nt,
        'absolute_lift': round(delta, 4),
        'relative_lift_pct': rel_lift,
        'z_statistic': round(z, 4),
        'p_value': round(p_value, 4),
        'confidence_interval': ci,
        'statistically_significant': bool(stat_sig),
        'practically_significant': bool(prac_sig),
        'required_sample_size': int(required_n),
        'recommendation': rec,
    }