import numpy as np
def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0) -> dict:
    num_judges=len(judge_scores)
    criteria_score = []
    sigma=[]
    for j in range(len(judge_scores[0])):
        s=sum(judge_scores[i][j] for i in range(num_judges))/num_judges
        criteria_score.append(s)
        sigma.append(np.sqrt(sum((judge_scores[i][j]-s)**2 for i in range(num_judges))/num_judges))
    weighted_score=float(np.array(criteria_weights)@(np.array(criteria_score).T))

    normalized_score = weighted_score/max_score
    pass_status = normalized_score>=passing_threshold
    sigma_bar = np.mean(sigma).astype(float)
    A = float(1-(2*sigma_bar/max_score))
    criteria_score=list(np.round(np.array(criteria_score),4))
    return {'weighted_score': round(weighted_score,4), 'normalized_score': round(normalized_score,4), 'criterion_scores': criteria_score, 'pass_status': pass_status, 'judge_agreement': round(A,4)}

