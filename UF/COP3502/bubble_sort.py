def bubble_sort(arr: list) -> None:
    for i in range(len(arr)):
        for j in range(1, len(arr) - i):
            if arr[j-1] > arr[j]:
                arr[j-1], arr[j] = arr[j], arr[j-1]


def main() -> None:
    size = int(input())
    input_list = []
    for _ in range(size):
        input_list.append(int(input()))

    bubble_sort(input_list)
    print(input_list)


if __name__ == "__main__":
    main()