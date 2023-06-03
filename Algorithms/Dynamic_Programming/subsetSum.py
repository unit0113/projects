def memoTargetSum(S, tgt):
    k = len(S)
    assert tgt >= 0
    ## Fill in base case for T[(i,j)] where i == k
    T = {} # Memo table initialized as empty dictionary
    for j in range(tgt+1):
        T[(k,j)] = j
    # your code here
    for i in range(k-1, -1, -1):
        for j in range(tgt+1):
            if j < S[i]:
                T[(i, j)] = T[(i+1, j)]
            else:
                T[(i, j)] = min(T[(i+1, j)], T[(i+1, j-S[i])])
    
    return T


def getBestTargetSum(S, tgt):
    k = len(S)
    assert tgt >= 0
    # your code here
    T = {} # Memo table initialized as empty dictionary
    for j in range(tgt+1):
        T[(k,j)] = []
    # your code here
    for i in range(k-1, -1, -1):
        for j in range(tgt+1):
            if j < S[i]:
                T[(i, j)] = T[(i+1, j)].copy()
            elif sum(T[(i+1, j)]) > sum(T[(i+1, j-S[i])]):
                T[(i, j)] = T[(i+1, j)].copy()
            else:
                T[(i, j)] = T[(i+1, j-S[i])].copy()
                T[(i, j)].append(S[i])
    
    return T[(0, tgt)]


def checkTgtSumRes(a, tgt,expected):
    if tgt != 347:
        a = sorted(a)
    res = getBestTargetSum(a, tgt)
    res = sorted(res)
    print('Your result:' , res)
    assert tgt - sum(res)  == expected, f'Your code returns result that sums up to {sum(res)}, expected was {expected}'
    i = 0
    j = 0
    n = len(a)
    m = len(res)
    if tgt == 347:
        a = sorted(a)
    while (i < n and j < m):
        if a[i] == res[j]: 
            j = j + 1
        i = i + 1
    assert j == m, 'Your result  {res} is not a subset of {a}'


a = [1, 2, 3, 4, 5, 10]
checkTgtSumRes(a, 15, 0)

a = [1, 8, 3, 4, 5, 12]
checkTgtSumRes(a, 26, 0)

a = [1, 10, 19, 18, 12, 11, 0, 9,  16, 17, 2, 7, 14, 29, 38, 45, 13, 26, 51, 82, 111, 124, 135, 189]
checkTgtSumRes(a, 347, 0)
checkTgtSumRes(a, 461, 0)
checkTgtSumRes(a, 462, 0)
checkTgtSumRes(a, 517, 0)
checkTgtSumRes(a, 975, 3)
