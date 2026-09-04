import math

def softmax(scores: list[float]) -> list[float]:
    output=[]
    calculated=0
    sums=0
    max_score=max(scores)
    sums = sum(math.exp(scores[t]-max_score) for t in range(len(scores)))
    for i in range(len(scores)):

        calculated=math.exp(scores[i]-max_score)/sums
        output.append(calculated)
    

    return output
    # Your code here
    pass