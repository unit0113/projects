def lssLength(a, i, j):
    max_ans = 1
    for i in range(len(a)):
        max_ans = max(max_ans, lssLengthEndingHere(a, i))
    return max_ans

def lssLengthEndingHere(a, curr):
    if curr == 0:
        return 1
    
    ans = 1
    for i in range(curr - 1, -1, -1):
        if abs(a[i] - a[curr]) <= 1:
            ans = max(ans, 1 + lssLengthEndingHere(a, i))
    return ans


print(lssLength([1, 4, 2, -2, 0, -1, 2, 3], 0, -1))