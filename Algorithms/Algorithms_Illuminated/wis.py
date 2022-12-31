def wis(arr: list[int]) -> int:
    memo = [0, arr[0]]
    for i in range(2, len(arr) + 1):
        memo.append(max(memo[i-1], memo[i-2] + arr[i-1]))

    return memo[-1]

array = [3, 2, 1, 6, 4, 5]
print(wis(array))