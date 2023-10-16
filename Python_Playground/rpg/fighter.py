import pygame
import random

from settings import POTION_RESTORE

FRAME_TIME = 0.075


class Fighter:
    def __init__(self, name, x, y, hp, strength, potions, sprites) -> None:
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.strength = strength
        self.max_potions = potions
        self.potions = potions
        self.sprites = sprites
        self.state = "Idle"
        self.frame_index = 0
        self.image = self.sprites[self.state][self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.frame_timer = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def can_use_potion(self) -> bool:
        return self.potions > 0 and self.hp < self.max_hp

    @property
    def should_heal(self) -> bool:
        return self.can_use_potion and self.hp + POTION_RESTORE <= self.max_hp

    def update(self, dt: float) -> None:
        self.frame_timer += dt
        if self.frame_timer > FRAME_TIME:
            self.frame_index += 1
            self.frame_timer = 0
            if self.frame_index >= len(self.sprites[self.state]):
                if self.state == "Death":
                    self.frame_index = len(self.sprites["Death"]) - 1
                    return

                self.frame_index = 0
                if self.state != "Idle":
                    self.state = "Idle"

    def idle(self) -> None:
        self.state = "Idle"
        self.frame_index = 0
        self.frame_timer = 0

    def attack(self, target) -> int:
        damage = self.strength + random.randint(-self.strength // 2, self.strength // 2)
        target.take_dmg(damage)
        self.state = "Attack"
        self.frame_index = 0
        self.frame_timer = 0

        return damage

    def take_dmg(self, dmg) -> None:
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.state = "Death"
        else:
            self.state = "Hurt"

        self.frame_index = 0
        self.frame_timer = 0

    def use_potion(self) -> int:
        amount_healed = min(POTION_RESTORE, self.max_hp - self.hp)
        self.potions -= 1
        self.hp = min(self.max_hp, self.hp + POTION_RESTORE)

        return amount_healed

    def draw(self, window) -> None:
        window.blit(self.sprites[self.state][self.frame_index], self.rect)
