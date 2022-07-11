from pakuri import pakuri


class pakudex:
    def __init__(self, capacity=20):
        self.max_capacity = capacity
        self.pakuris = []

    @property
    def is_full(self):
        return len(self.pakuris) >= self.max_capacity

    def find_paku(self, species):
        for paku in self.pakuris:
            if species == paku.get_species():
                return paku

        return None

    def get_size(self):
        return len(self.pakuris)

    def get_capacity(self):
        return self.max_capacity

    def get_species_array(self):
        if self.pakuris:
            return [paku.get_species() for paku in self.pakuris]

    def get_stats(self, species):
        paku = self.find_paku(species)
        if paku:
            return [paku.get_attack(), paku.get_defense(), paku.get_speed()]

        # If not found
        return None

    def sort_pakuri(self):
        self.pakuris.sort(key=lambda x: x.get_species())

    def add_pakuri(self, species):
        # If dex is full
        if self.is_full or self.find_paku(species):
            return False
        
        self.pakuris.append(pakuri(species))
        return True

    def evolve_species(self, species):
        paku = self.find_paku(species)
        if paku:
            paku.evolve()
            return True
        else:
            return False
