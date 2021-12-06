import sys
import csv


def maxrepeats(dna, STR):

    # Iteration values
    i = 0
    j = len(STR)

    # Counter
    maxrepeat = 0

    for k in range(len(dna)):
        if dna[i:j] == STR:
            temp = 0
            # check for repeats
            while dna[i:j] == STR:
                temp += 1
                # check next block
                i += len(STR)
                j += len(STR)
                # update max if bigger
                if temp > maxrepeat:
                    maxrepeat = temp
        # Move to next index
        else:
            i += 1
            j += 1

    return maxrepeat


def match(reader, maxreps):

    # Read info in csv
    for row in reader:
        name = row[0]
        values = [int(value) for value in row[1:]]

        # Check for match
        if values == maxreps:
            print(name)
            return

    # If no match
    print("No match")


def main():

    # Ensure correct usage
    if len(sys.argv) != 3:
        sys.exit("Usage: python dna.py data.csv sequence.txt")

    # Open file
    with open(sys.argv[1], "r") as STRfile:
        reader = csv.reader(STRfile)

        # Get list of possible STRs
        STRs = next(reader)[1:]

        # Read DNA sequence
        dnafile = open(sys.argv[2], "r")
        dna = dnafile.read()

        # Count repeats
        maxreps = [maxrepeats(dna, STR) for STR in STRs]

        # Check for matches
        match(reader, maxreps)


if __name__ == "__main__":
    main()