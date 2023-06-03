def lisLength(a, i, j):
    max_ans = 1
    for i in range(len(a)):
        max_ans = max(max_ans, lisLengthEndingHere(a, i))
    return max_ans

def lisLengthEndingHere(a, curr):
    if curr == 0:
        return 1
    
    ans = 1
    for i in range(curr - 1, -1, -1):
        if a[i] < a[curr]:
            ans = max(ans, 1 + lisLengthEndingHere(a, i))
    return ans




print(lisLength([10, 22, 9, 33, 21, 50, 41, 60], 0, -1))