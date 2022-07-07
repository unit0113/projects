def hex_char_decode(digit):
    return int(digit, 16)


def hex_string_decode(hex):
    return int(hex, 16)


def binary_string_decode(binary):
    return int(binary, 2)


def binary_to_hex(binary):
    return hex(int(binary, 2)).replace('0x', '').upper()


class NumericConverter:
    def __init__(self):
        self.menu()


    def menu(self):
        # Print Menu
        print('Decoding Menu')
        print('-------------')
        print('1. Decode hexadecimal')
        print('2. Decode binary')
        print('3. Convert binary to hexadecimal')
        print('4. Quit')
        print()

        # Get valid input
        option = -1
        while option < 0 or option > 4:
            try:
                option = int(input('Please enter an option: '))
            except:
                continue

            if option < 0 or option > 4:
                print()
                print('Error! Invalid input.')
                print()

        # Exit
        if option == 4:
            print('Goodbye!')
            quit()

        # Call appropriate functions
        if option == 1:
            self.decode_hex()

        elif option == 2:
            self.decode_binary()

        elif option == 3:
            self.binary_to_hex()

    def decode_hex(self):
        """Convert hex to decimal
        """
        hex = input('Please enter the numeric string to convert: ').replace('0x', '').lower()
        print(f'Result: {hex_string_decode(hex)}')
        print()
        self.menu()

    def decode_binary(self):
        """Convert binary to decimal
        """
        binary = input('Please enter the numeric string to convert: ')
        print(f'Result: {binary_string_decode(binary)}')
        print()
        self.menu()

    def binary_to_hex(self):
        """Convert binary to hex
        """
        binary = input('Please enter the numeric string to convert: ')
        print(f'Result: {binary_to_hex(binary)}')
        print()
        self.menu()


def main():
    nc = NumericConverter()


if __name__ == "__main__":
    main()
