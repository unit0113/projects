import re
import sys


def main():
    print(count(input("Text: ")))


def count(s):
    expression = re.compile(r"(?:^|\W)um(?:$|\W)", re.I)
    return len(re.findall(expression, s))


if __name__ == "__main__":
    main()
