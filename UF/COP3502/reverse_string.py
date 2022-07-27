def reverse(input_str: str) -> str:
    reversed_str = ""
    for i in range(len(input_str)-1, -1, -1):
        reversed_str += input_str[i]

    return reversed_str


def main() -> None:
    input_str = input()
    print(f'Reverse of "{input_str}" is "{reverse(input_str)}".')


if __name__ == "__main__":
    main()