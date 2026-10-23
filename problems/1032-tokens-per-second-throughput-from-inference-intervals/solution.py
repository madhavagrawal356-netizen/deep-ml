def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    step_tps = []
    cumulative_tps = []
    

    cum_t = 0
    cum_s = 0
    if not intervals:
        return {'step_tps': step_tps, 'cumulative_tps': cumulative_tps, 'overall_tps': 0.0}
    for t,s in intervals:
        if t<0:
            raise ValueError
        elif s<=0:
            raise ValueError
        else:
            cum_t+=t 
            cum_s+=s 
            step_tps.append(round(t/s, 2))
            cumulative_tps.append(round(cum_t/cum_s,2)) 
    return {'step_tps': step_tps, 'cumulative_tps': cumulative_tps, 'overall_tps': cum_t/cum_s}


