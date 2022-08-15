import sys
from os.path import exists, splitext
from PIL import Image, ImageOps


NUM_ARGS = 3
VALID_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def main():
    # Check program input
    if len(sys.argv) < NUM_ARGS:
        sys.exit("Too few command-line arguments")

    elif len(sys.argv) > NUM_ARGS:
        sys.exit("Too many command-line arguments")

    ext1 = splitext(sys.argv[1])[-1]
    if ext1 not in VALID_FILE_EXTENSIONS:
        sys.exit("Invalid input")

    ext2 = splitext(sys.argv[2])[-1]
    if ext2 not in VALID_FILE_EXTENSIONS:
        sys.exit("Invalid input")

    if ext1 != ext2:
        sys.exit("Input and output have different extensions")

    # Check if file exists
    if not exists(sys.argv[1]):
        sys.exit("Input does not exist")

    # Paste shirt
    shirt = Image.open("shirt.png")
    muppet = Image.open(sys.argv[1])
    muppet = ImageOps.fit(muppet, shirt.size)

    muppet.paste(shirt, shirt)
    muppet.save(sys.argv[2])


if __name__ == "__main__":
    main()