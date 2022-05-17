import random
from sre_parse import SPECIAL_CHARS
from string import ascii_lowercase, ascii_uppercase


SPECIAL_CHARS = list(SPECIAL_CHARS)
UPPER_CASE = list(ascii_uppercase)
LOWER_CASE = list(ascii_lowercase)
NUMBERS = [str(num) for num in range(0,10)]


def main():
    num_spec_chars = int(input('Enter the minimum number of special characters: '))
    num_upper_case = int(input('Enter the minimum number of capital letters: '))
    num_numbers = int(input('Enter the minimum number of numbers: '))
    pw_length = int(input('Enter the desired password length: '))

    password = []

    for _ in range(num_spec_chars):
        password.append(random.choice(SPECIAL_CHARS))

    for _ in range(num_upper_case):
        password.append(random.choice(UPPER_CASE))

    for _ in range(num_numbers):
        password.append(random.choice(NUMBERS))

    everything = SPECIAL_CHARS + UPPER_CASE + LOWER_CASE + NUMBERS

    while len(password) < pw_length:
        password.append(random.choice(everything))

    random.shuffle(password)
    password = ''.join(password)
    print(password)


if __name__ == "__main__":
    main()