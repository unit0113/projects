def main():
    str_in = input("Input: ")
    print(f'Output: {shorten(str_in)}')


def shorten(str_in):
    vowels = 'aeiou'
    str_out = ''
    for char in str_in:
        if char.lower() in vowels:
            continue
        str_out += char

    return str_out


if __name__ == "__main__":
    main()
