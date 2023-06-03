import math

def targetSum(S, i,  tgt):
    # your code here
    if tgt < 0:
        return math.inf
    elif i > len(S) - 1:
        return tgt
    else:
        return min(targetSum(S, i+1, tgt), targetSum(S, i+1, tgt-S[i]))