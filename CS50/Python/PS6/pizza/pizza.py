import sys
from os.path import exists
import csv


def read_menu(file_name: str) -> list:
    menu = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            menu.append(row)

    return menu


def print_menu(menu: list) -> None:
    width = len(menu[0][0]) + 3
    print_divider(width)
    print_row(menu[0], width)
    print_thicc_divider(width)
    for row in menu[1:]:
        print_row(row, width)
        print_divider(width)


def print_divider(width: int) -> None:
    print(f"+-{'-'*width}+---------+---------+")


def print_thicc_divider(width: int) -> None:
    print(f"+={'='*width}+=========+=========+")


def print_row(row: list, width:int) -> None:
    print(f"| {row[0]:<{width}}| {row[1]:<8}| {row[2]:<8}|")


def main():
    # Check program input
    if len(sys.argv) < 2:
        sys.exit("Too few command-line arguments")

    elif len(sys.argv) > 2:
        sys.exit("Too many command-line arguments")

    if not sys.argv[1].endswith(".csv"):
        sys.exit("Not a Python file")

    # Check if file exists
    if not exists(sys.argv[1]):
        sys.exit("File does not exist")

    # Get and print menu
    menu = read_menu(sys.argv[1])
    print_menu(menu)

if __name__ == "__main__":
    main()
