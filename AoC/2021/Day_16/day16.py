with open(r'AoC\2021\Day_16\test_input.txt', 'r') as file:
    bits = file.readline()


def parse_subpacket(bits_remainder):
    result_string = ''
    index = 0
    while index + 5 < len(bits_remainder):
        result_string += bits_remainder[index+1:index+5]
        index += 5

    return result_string


bits = bin(int(bits, 16))[2:].zfill(len(bits)*4)
version = int(bits[:3], 2)
type_ID = int(bits[3:6], 2)
length_type_id = int(bits[6])
subpacket_length_bits = 15 if length_type_id == 0 else 11
subpacket_length = int(bits[7:7+subpacket_length_bits], 2)
bits_remainder = bits[7+subpacket_length_bits:]

result_string = parse_subpacket(bits_remainder)

if type_ID == 4:
    result = int(result_string, 2)

print(version)
print(type_ID)
print(length_type_id)
print(subpacket_length_bits)
print(subpacket_length)
print(bits_remainder)