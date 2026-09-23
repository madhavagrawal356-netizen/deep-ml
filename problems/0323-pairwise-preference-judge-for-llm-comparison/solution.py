import numpy as np
def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    if comparisons==[]:
        return {'results': [], 'win_rate_a': 0.0, 'win_rate_b': 0.0, 'tie_rate': 0.0, 'avg_margin': 0.0}
    results=[]
    a_wins=0
    b_wins=0
    ties = 0
    n = len(comparisons)
    weights = np.array([values for key, values in criteria_weights.items()])
    sa=0
    sb=0
    margins=[]
    for item in comparisons:
        a_scores=[values for key,values in item['scores_a'].items()]
        sa = np.array(a_scores)@weights
        b_scores=[values for key, values in item['scores_b'].items()]
        sb = np.array(b_scores)@weights
        margin=(sa-sb)/(weights.sum())
        margins.append(round(abs(margin),4))
        if margin>tie_threshold:
            winner='A'
            a_wins+=1
        elif margin<-tie_threshold:
            winner='B'
            b_wins+=1
        else:
            winner='tie'
            ties+=1
        results.append({'id':item['id'], 'winner':winner,
        'margin':round(float(abs(margin)),4)})
    win_a = a_wins/n 
    win_b = b_wins/n 
    tiess = ties/n 
    avg_margin = np.mean(margins)
    return {'results': results, 'win_rate_a':round(win_a,4), 'win_rate_b':round(win_b,4), 'tie_rate': round(tiess,4), 'avg_margin': round(float(avg_margin),4)}

    
