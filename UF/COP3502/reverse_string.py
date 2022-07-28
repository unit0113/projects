def reverse(input_str: str) -> str:
    
    if len(input_str) <= 1:
        return input_str

    return input_str[-1] + reverse(input_str[:-1])


def main() -> None:
    input_str = input()
    print(f'Reverse of "{input_str}" is "{reverse(input_str)}".')


if __name__ == "__main__":
    main()