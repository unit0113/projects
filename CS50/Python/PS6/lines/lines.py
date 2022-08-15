import sys
from os.path import exists


def main():
    # Check program input
    if len(sys.argv) < 2:
        sys.exit("Too few command-line arguments")

    elif len(sys.argv) > 2:
        sys.exit("Too many command-line arguments")

    if not sys.argv[1].endswith(".py"):
        sys.exit("Not a Python file")

    # Check if file exists
    if not exists(sys.argv[1]):
        sys.exit("File does not exist")

    # Scan file
    line_count = 0
    with open(sys.argv[1], 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line and not line.startswith('#'):
                line_count += 1

    print(line_count)

if __name__ == "__main__":
    main()
