import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
            def sigmoid(x):
                return 1/(1+np.exp(-x))
            chosen_log_ratios = np.array(log_probs_chosen_policy)-np.array(log_probs_chosen_ref)
            rejected_log_ratios = np.array(log_probs_rejected_policy)-np.array(log_probs_rejected_ref)
            logits = beta*(chosen_log_ratios-rejected_log_ratios)
            loss = np.mean(-np.log(sigmoid(logits)), axis=0)
            chosen = beta*chosen_log_ratios
            rejected = beta*rejected_log_ratios
            return {'loss': np.round(loss,4).astype(float), 'chosen_rewards': np.round(chosen,4).tolist(), 'rejected_rewards': np.round(rejected,4).tolist()}

            