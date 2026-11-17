import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    thresholds = np.unique(y_scores)[::-1]
    precisions=[]
    recall=[]
    for threshold in thresholds:
        TP=0
        FP=0
        AP=0
        y_hat= y_scores.copy()
        for i in range(len(y_hat)):
            y_hat[i]=int(y_hat[i]>=threshold)
            if (y_hat[i]==1 and y_true[i]==0):
                FP+=1 
            elif (y_hat[i]==1 and y_true[i]==1):
                TP+=1 
            if y_true[i]==1:
                AP+=1
        if (TP==0 and FP==0):
            precisions.append(1.0)
        else:
            precisions.append(TP/(TP+FP))
        if AP==0:
            recall.append(0.0)
        else:
            recall.append(TP/AP)
    return precisions, recall, thresholds
            
        