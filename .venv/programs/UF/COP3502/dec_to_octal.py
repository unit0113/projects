def dec_to_octal(dec):
    return oct(int(dec)).lstrip('0o')

if __name__ == '__main__':
    decimal = input()
    print(dec_to_octal(decimal))