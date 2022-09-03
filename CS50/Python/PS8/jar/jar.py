class Jar:
    def __init__(self, capacity=12):
        self.cookies = 0
        if isinstance(capacity, int) and capacity >= 0:
            self._capacity = capacity
        else:
            raise ValueError

    def __str__(self):
        return "🍪" * self.cookies

    def deposit(self, n):
        if self.cookies + n > self.capacity:
            raise ValueError
        else:
            self.cookies += n

    def withdraw(self, n):
        if self.cookies - n < 0:
            raise ValueError
        else:
            self.cookies -= n

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return self.cookies