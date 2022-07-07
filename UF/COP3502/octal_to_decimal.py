def octal_string_decode(oct):
    return int(oct.lstrip('0'), 8)


if __name__ == '__main__':
    octal = input()
    print(octal_string_decode(octal))