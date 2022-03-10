class dyn_array():
    def __init__(self, *argv):
        self.array = []
        self.expansion_factor = 2
        self.array_length = 2
        self.length = 0
        for arg in argv:
            self.array.add(arg)

    def expand(self):
        self.array_length *= self.expansion_factor

    def add(self, item):
        if self.length >= self.array_length:
                self.expand()
        self.array.append(item)
        self.length += 1