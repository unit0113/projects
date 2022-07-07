phrase1 = 'HIEROGLYPHOLOGY'
phrase2 = 'MICHAELANGELO'

def lcs(phrase1, phrase2):
    memo = [[''] * (len(phrase2)) for _ in range(len(phrase1))]

    for i in range(len(phrase1)):
        for j in range(len(phrase2)):
            if phrase1[i] == phrase2[j]:
                if i == 0 or j == 0:
                    memo[i][j] = phrase1[i]
                else:
                    memo[i][j] = memo[i-1][j-1] + phrase1[i]

            else:
                memo[i][j] = max(memo[i-1][j], memo[i][j-1], key=len)

    return memo[-1][-1]


print(lcs(phrase1, phrase2))