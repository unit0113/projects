import re
import sys


def main():
    print(validate(input("IPv4 Address: ")))


def validate(ip):
    if match := re.search(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$", ip):
        blk1, blk2, blk3, blk4 = match.groups()
        if valid_num(blk1) and valid_num(blk2) and valid_num(blk3) and valid_num(blk4):
            return True

    return False


def valid_num(num):
    return 0 <= int(num) <= 255


if __name__ == "__main__":
    main()