from cs50 import get_int

# Get height from user, between 1-8 inclusive
height = 0
while height < 1 or height > 8:
    height = get_int("Height: ")

# Print half pyramid
for i in range(height):

    # Print spaces
    print(" " * (height - i - 1), end='')

    # Print blocks
    print("#" * (i + 1), end='')

    i += 1
    print()