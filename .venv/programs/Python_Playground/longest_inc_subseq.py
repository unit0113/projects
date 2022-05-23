phrase = 'CARBOHYDRATE'


def longest_inc_subseq(phrase):
    memo = [[] for _ in range(len(phrase))]

    # First increasing subsequence is the first element in A
    memo[0].append(phrase[0])

    for i in range(1, len(phrase)):
        for j in range(0, i):
            # Check if new larger subsequence found
            if (phrase[j] < phrase[i]) and (len(memo[i]) < len(memo[j])):
                # Wipe current list
                memo[i] = []
                # Copy memo[j] into memo[i]
                memo[i].extend(memo[j])

        # Add current element onto current memo
        memo[i].append(phrase[i])

    return ''.join(max(memo, key=len))
    

print(longest_inc_subseq(phrase))