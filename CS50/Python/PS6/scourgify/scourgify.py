import sys
from os.path import exists
from csv import DictReader, DictWriter


NUM_ARGS = 3


def main():
    # Check program input
    if len(sys.argv) < NUM_ARGS:
        sys.exit("Too few command-line arguments")

    elif len(sys.argv) > NUM_ARGS:
        sys.exit("Too many command-line arguments")

    if not sys.argv[1].endswith(".csv"):
        sys.exit("Not a csv file")

    if not sys.argv[2].endswith(".csv"):
        sys.exit("Not a csv file")

    # Check if file exists
    if not exists(sys.argv[1]):
        sys.exit(f"Could not read {sys.argv[1]}")

    # Read file
    with open(sys.argv[1], 'r') as in_file:
        reader = DictReader(in_file)
        students = []
        for row in reader:
            last, first = row["name"].split(', ')
            students.append({"first": first, "last": last, "house": row["house"]})

    # Write file
    with open(sys.argv[2], 'w') as out_file:
        writer = DictWriter(out_file, fieldnames=["first","last","house"])
        writer.writeheader()
        for row in students:
            writer.writerow(row)


if __name__ == "__main__":
    main()
