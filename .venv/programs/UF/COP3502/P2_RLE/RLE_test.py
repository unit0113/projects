import unittest
from rle_program import *

class TestRLE(unittest.TestCase):

    def test_to_hex_string(self):
        input = [3, 15, 6, 4]
        self.assertEqual(to_hex_string(input), '3f64')


    def test_count_runs(self):
        input = [15] * 3 + [4] * 6
        self.assertEqual(count_runs(input), 2)
        input = [15] * 3 + [4] * 6 + [15] * 3
        self.assertEqual(count_runs(input), 3)


    def test_encode_RLE(self):
        input = [15] * 3 + [4] * 6
        self.assertEqual(encode_rle(input), [3, 15, 6, 4])
        input = [15] * 3 + [4] * 6 + [15] * 3
        self.assertEqual(encode_rle(input), [3, 15, 6, 4, 3, 15])
        input = [15] * 3 + [4] * 6 + [15] * 3 + [9]
        self.assertEqual(encode_rle(input), [3, 15, 6, 4, 3, 15, 1, 9])

    
    def test_get_decoded_length(self):
        input = [15] * 3 + [4] * 6
        rle_input = encode_rle(input)
        self.assertEqual(get_decoded_length(rle_input), 9)
        input = [15] * 3 + [4] * 6 + [15] * 3
        rle_input = encode_rle(input)
        self.assertEqual(get_decoded_length(rle_input), 12)
        input = [15] * 3 + [4] * 6 + [15] * 3 + [9]
        rle_input = encode_rle(input)
        self.assertEqual(get_decoded_length(rle_input), 13)

    
    def test_decode_RLE(self):
        input = [3, 15, 6, 4]
        self.assertEqual(decode_rle(input), [15] * 3 + [4] * 6)
        input = [3, 15, 6, 4, 3, 15]
        self.assertEqual(decode_rle(input), [15] * 3 + [4] * 6 + [15] * 3)
        input = [3, 15, 6, 4, 3, 15, 1, 9]
        self.assertEqual(decode_rle(input), [15] * 3 + [4] * 6 + [15] * 3 + [9])


    def test_string_to_data(self):
        input = '3f64'
        self.assertEqual(string_to_data(input), [3, 15, 6, 4])


    def test_to_rle_string(self):
        input = [15, 15, 6, 4]
        self.assertEqual(to_rle_string(input), '15f:64')


    def test_string_to_rle(self):
        input = '15f:64'
        self.assertEqual(string_to_rle(input), [15, 15, 6, 4])


    def test_thruput(self):
        input = 'eefffffffffffffffffffffffffffffffffff'
        answer_1 = '2e:15f:15f:5f'
        self.assertEqual(to_rle_string(encode_rle(string_to_data(input))), answer_1)



if __name__ == '__main__':
    unittest.main()