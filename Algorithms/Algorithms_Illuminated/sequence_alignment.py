



MATCH = 1
MISMATCH = -1
GAP = -2


# Global alignment
def nw(seq1: str, seq2: str) -> None:
    m = len(seq1) + 1
    n = len(seq2) + 1
    memo = [[0] * (n) for _ in range(m)]

    for index in range(m):
        memo[index][0] = index * GAP

    for index in range(n):
        memo[0][index] = index * GAP

    for i in range(1, m):
        for j in range(1, n):
            alpha = MATCH if seq1[i-1] == seq2[j-1] else MISMATCH
            memo[i][j] = max(memo[i-1][j-1] + alpha, memo[i-1][j] + GAP, memo[i][j-1] + GAP)

    match_value =  memo[-1][-1]

    # Traceback
    aligned_seq1 = ""
    aligned_seq2 = ""

    i = len(seq1)
    j = len(seq2)
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif memo[i][j] == memo[i-1][j] + GAP:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = '_' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '_' + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1

    print(f"Alignment score: {match_value}")
    print(aligned_seq1)
    print(aligned_seq2)


nw("ATGCT", "AGCT")
