import math
from bitarray import bitarray
import mmh3

class Bloom_filter:
    def __init__(self, item_count: int, false_positive_prob: float) -> None:
        self.size = int(-(item_count * math.log(false_positive_prob))/(math.log(2)**2))
        self.hash_count = int((self.size/item_count) * math.log(2))

        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)


    def lookup(self, item) -> bool:
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            if self.bit_array[index] == False:
                return False

        return True

    def insert(self, item) -> bool:
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = True


bf = Bloom_filter(10, 0.05)
bf.insert("Hello")
print(bf.lookup("Hello"))
print(bf.lookup("Hi"))
