from random import randint

from game_data import MONSTER_DATA, ATTACK_DATA


class Monster:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.paused = False

        # Stats
        self.element = MONSTER_DATA[name]['stats']['element']
        self.base_stats = MONSTER_DATA[name]['stats']
        self.health = self.base_stats['max_health'] * self.level
        self.energy = self.base_stats['max_energy'] * self.level
        self.initiative = 0
        self.abilities = MONSTER_DATA[name]['abilities']

        # XP
        self.xp = randint(0, 1000)
        self.level_up = self.level * 150

    def __repr__(self):
        return f'Monster: {self.name}, lvl: {self.level}'

    def get_stat(self, stat):
        return self.base_stats[stat] * self.level
    
    def get_stats(self):
        return {
            'Health': self.get_stat('max_health'),
            'Energy': self.get_stat('max_energy'),
            'Attack': self.get_stat('attack'),
            'Defense': self.get_stat('defense'),
            'Speed': self.get_stat('speed'),
            'Recovery': self.get_stat('recovery'),
        }
    
    def get_abilities(self, all=True):
        if all:
            return [ability for lvl, ability in self.abilities.items() if self.level >= lvl]
        else:
            return [ability for lvl, ability in self.abilities.items() if self.level >= lvl and ATTACK_DATA[ability]['cost'] <= self.energy]
    
    def get_info(self):
        return (
            (self.health, self.get_stat('max_health')),
            (self.energy, self.get_stat('max_energy')),
            (self.initiative, 100)
                )
    
    def update(self, dt):
        if not self.paused:
            self.initiative += self.get_stat('speed') * dt