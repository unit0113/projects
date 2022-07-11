class pakuri:
    def __init__(self, name):
        self._name = name
        self._attack = len(name) * 7 + 9
        self._defense = len(name) * 5 + 17
        self._speed = len(name) * 6 + 13

    def get_species(self):
        return self._name

    def get_attack(self):
        return self._attack

    def get_defense(self):
        return self._defense

    def get_speed(self):
        return self._speed

    def set_attack(self, new_attack):
        self._attack = new_attack

    def evolve(self):
        self._attack *= 2
        self._defense *= 4
        self._speed *= 3
