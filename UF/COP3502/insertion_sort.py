def insertion_sort(arr: list) -> None:
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j-1]:
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1


def main() -> None:
    size = int(input())
    input_list = []
    for _ in range(size):
        input_list.append(int(input()))

    insertion_sort(input_list)
    print(input_list)


if __name__ == "__main__":
    main()