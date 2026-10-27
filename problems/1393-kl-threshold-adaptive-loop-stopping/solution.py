import torch


def adaptive_loop_exit(step_fn, s0, coda, max_loops, kl_threshold):
    s = s0
    logp_prev = None
    kl_history = []
    loops_used = 0

    for loop in range(max_loops):
        s = step_fn(s)
        logp = torch.log_softmax(coda(s), dim=-1)
        loops_used = loop + 1

        if logp_prev is not None:
            kl = (logp_prev.exp() * (logp_prev - logp)).sum(dim=-1).mean()
            kl_history.append(kl.item())
            if kl.item() < kl_threshold:
                break

        logp_prev = logp

    return s, loops_used, kl_history


