def target_string_count(main_str: str, sub_str: str) -> int:
    if len(main_str) < len(sub_str):
        return 0
    
    if main_str[:len(sub_str)] == sub_str:
        return 1 + target_string_count(main_str[1:], sub_str)

    else:
        return target_string_count(main_str[1:], sub_str)


def main() -> None:
    main_str, target_str = input().split()
    print(target_string_count(main_str, target_str))


if __name__ == "__main__":
    main()