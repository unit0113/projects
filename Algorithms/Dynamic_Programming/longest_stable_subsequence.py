def memoizeLSS(a):
    T = {} # Initialize the memo table to empty dictionary
    # Now populate the entries for the base case 
    n = len(a)
    for j in range(-1, n):
        T[(n, j)] = 0 # i = n and j 
    # Now fill out the table : figure out the two nested for loops
    # It is important to also figure out the order in which you iterate the indices i and j
    # Use the recurrence structure itself as a guide: see for instance that T[(i,j)] will depend on T[(i+1, j)]
    # your code here
    
    #for i in range(1, n):
        #for j in range(i):

    for i in  range(n):
        for j in range(-1, n):
            T[(i, j)] = 0
    
    n = len(a)
    lis = [1]*n
    for i in range(1, n):
        for j in range(0, i):
            if abs(a[i] - a[j]) <= 1 and lis[i] < lis[j] + 1:
                lis[i] = lis[j]+1

    T[0, -1] = max(lis)
    
    return T


def computeLSS(a):
    # your code here
    n = len(a)
    sequences = [[] for _ in range(n)]
    sequences[0].append(a[0])
    
    for i in range(1, n):
        for j in range(0, i):
            if abs(a[i] - a[j]) <= 1 and len(sequences[i]) < len(sequences[j]) + 1:
                sequences[i] = sequences[j].copy()
                
        sequences[i].append(a[i])
        
    return max(sequences, key=len)


def checkMemoTableBaseCase(a, T):
    n = len(a)
    for j in range(-1, n):
        assert T[(n, j)] == 0, f'entry for {(n,j)} is not zero as expected'


def checkMemoTableHasEntries(a, T):
    for i in range(len(a)+1):
        for j in range(i):
            assert (i, j) in T, f'entry for {(i,j)} not in memo table'


a = [1, 4, 2, -2, 0, -1, 2, 3]
T = memoizeLSS(a)
print(T)
print(T[0, -1])
checkMemoTableBaseCase(a, T)
checkMemoTableHasEntries(a, T)