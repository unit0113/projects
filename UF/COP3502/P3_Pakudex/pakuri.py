class pakuri:
    def __init__(self, name):
        self._name = name
        self._attack = len(name) * 7 + 9
        self._defense = len(name) * 5 + 17
        self._speed = len(name) * 6 + 13

    def get_species(self):
        # Getter for species name
        return self._name

    def get_attack(self):
        # Getter for attack value
        return self._attack

    def get_defense(self):
        # Getter for defense value
        return self._defense

    def get_speed(self):
        # Getter for speed value
        return self._speed

    def set_attack(self, new_attack):
        # Setter for attack value
        self._attack = new_attack

    def evolve(self):
        # It's bigger, it's badder...
        self._attack *= 2
        self._defense *= 4
        self._speed *= 3
