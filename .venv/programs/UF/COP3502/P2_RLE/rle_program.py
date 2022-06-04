from console_gfx import ConsoleGfx
import os
import sys


def to_hex_string(data):
    """Translates data (RLE or raw) a hexadecimal string (without delimiters)

    Args:
        data (list): RLE or raw data

    Returns:
        string: Converted hex data
    """

    # Remove '0x' for storage
    hex_list = [hex(item).replace('0x', '') for item in data]
    return ''.join(hex_list)


def count_runs(flat_data):
    """Returns number of runs of data in an image data set; double this result for length of encoded (RLE) list

    Args:
        flat_data (list): Image data

    Returns:
        int: Number of runs
    """

    # Base case
    if len(flat_data) <= 1:
        return len(flat_data)

    last_seen = flat_data[0]
    count = 1
    current_run = 1
    for item in flat_data[1:]:
        # Find start of new run
        if item != last_seen:
            last_seen = item
            count += 1
            current_run = 1
        
        else:
            current_run += 1

        # Account for runs longer than 15
        if current_run == 15:
            current_run = 0
            count += 1        

    return count


def encode_rle(flat_data):
    """Returns encoding (in RLE) of the raw data passed in

    Args:
        flat_data (list): Flat data

    Returns:
        list: RLE data
    """

    result = []
    last_seen = flat_data[0]
    count = 1
    for item in flat_data[1:]:
        # Get count
        if item == last_seen:
            count += 1

        # Start new run
        else:
            result += [count, last_seen]
            last_seen = item
            count = 1
        
        if count == 15:
            result += [count, last_seen]
            count = 0

    # Catch last item
    result += [count, last_seen]

    return result


def get_decoded_length(rle_data):
    """Returns decompressed size RLE data; used to generate flat data from RLE encoding

    Args:
        rle_data (list): RLE encoded data

    Returns:
        int: Length of data
    """

    # Sum every other
    return sum([count for count in rle_data[::2]])


def decode_rle(rle_data):
    """Returns the decoded data set from RLE encoded data

    Args:
        rle_data (list): RLE data

    Returns:
        list: Flat data
    """

    # Step by two
    result = []
    for index in range(0, len(rle_data), 2):
        result += [rle_data[index+1]] * rle_data[index]

    return result


def string_to_data(data_string):
    """Translates a string in hexadecimal format into byte data (can be raw or RLE)

    Args:
        data_string (string): Hex input string

    Returns:
        list: RLE data
    """

    return [int(item, 16) for item in list(data_string)]


def to_rle_string(rle_data):
    """Translates RLE data into a human-readable representation

    Args:
        rle_data (list): RLE data

    Returns:
        string: Human readable hex run data
    """

    result = []
    for run_length, run_value in zip(rle_data[::2], rle_data[1::2]):
        # First number is decimal representation of hex number, max of 15
        while run_length > 15:
            result.append('15' + hex(run_value).replace('0x', ''))
            run_length -= 15

        result.append(str(run_length) + hex(run_value).replace('0x', ''))

    return ':'.join(result)


def string_to_rle(rle_string):
    """Translates a string in human-readable RLE format (with delimiters) into RLE byte data

    Args:
        rle_string (string): Human readable hex run data

    Returns:
        list: RLE data
    """
    
    rle_string_list = rle_string.split(':')

    result = []
    for run in rle_string_list:
        # Last value into run_value, rest into run_length
        *run_length, run_value = run
        result += [int(''.join(run_length)), int(run_value, 16)]

    return result


class RLE:
    def __init__(self):
        self.data = None
        print('Welcome to the RLE image encoder!')
        print()
        print('Displaying Spectrum Image:')
        ConsoleGfx.display_image(ConsoleGfx.test_rainbow)


    def menu(self):
        # Print Menu
        print('RLE Menu')
        print('-' * 8)
        print('0. Exit')
        print('1. Load File')
        print('2. Load Test Image')
        print('3. Read RLE String')
        print('4. Read RLE Hex String')
        print('5. Read Data Hex String')
        print('6. Display Image')
        print('7. Display RLE String')
        print('8. Display Hex RLE Data')
        print('9. Display Hex Flat Data')
        print()

        # Get valid input
        option = -1
        while option < 0 or option > 9:
            try:
                option = int(input('Select a Menu Option: '))
            except:
                continue

            if option < 0 or option > 9:
                print()
                print('Error! Invalid input.')
                print()

        # Exit
        if option == 0:
            quit()
        
        # Call appropriate functions
        if option == 1:
            self.load_file()

        elif option ==2:
            self.load_test_image()

        elif option == 3:
            self.read_rle_string()

        elif option == 4:
            self.read_rle_hex_string()

        elif option == 5:
            self.read_data_hex_string()

        elif option == 6:
            self.display_image()

        elif option == 7:
            self.display_rle_string()

        elif option == 8:
            self.display_hex_rle_data()

        elif option == 9:
            self.display_hex_flat_data()


    def load_file(self):
        """Load data from file
        """
        file_name = input('Enter name of file to load: ')
        file_path = os.path.join(sys.path[0], file_name)
        self.data = ConsoleGfx.load_file(file_path)
        print()
        self.menu()


    def load_test_image(self):
        """Load test image data
        """
        self.data = ConsoleGfx.test_image
        print('Test image data loaded.')
        print()
        self.menu()


    def read_rle_string(self):
        """Decode RLE string and store as flat data
        """
        rle_string = input('Enter an RLE string to be decoded: ')
        rle_data = string_to_rle(rle_string)
        self.data = decode_rle(rle_data)
        print()
        self.menu()


    def read_rle_hex_string(self):
        """Decode hex RLE string and store as flat data
        """
        rle_string = input('Enter the hex string holding RLE data: ')
        rle_data = string_to_data(rle_string)
        self.data = decode_rle(rle_data)
        print()
        self.menu()


    def read_data_hex_string(self):
        """Convert hex flat data and store
        """
        rle_string = input('Enter the hex string holding flat data: ')
        self.data = string_to_data(rle_string)
        print()
        self.menu()


    def display_image(self):
        """Displays stored image data
        """
        print('Displaying image...')
        ConsoleGfx.display_image(self.data)
        print()
        self.menu()


    def display_rle_string(self):
        """Display currently stored flat data as RLE string
        """
        encoded_rle = encode_rle(self.data)
        rle_output = to_rle_string(encoded_rle)
        print(f'RLE representation: {rle_output}')
        print()
        self.menu()


    def display_hex_rle_data(self):
        """Display currently stored flat data as hex RLE string
        """
        encoded_rle = encode_rle(self.data)
        hex_output = to_hex_string(encoded_rle)
        print(f'RLE hex values: {hex_output}')
        print()
        self.menu()


    def display_hex_flat_data(self):
        """Display currently stored flat data as hex string
        """
        hex_output = to_hex_string(self.data)
        print(f'Flat hex values: {hex_output}')
        print()
        self.menu()


def main():
    rle = RLE()
    rle.menu()


if __name__ == "__main__":
    main()