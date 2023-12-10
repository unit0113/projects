import pygame

from .ship import Ship


class Enemy(Ship):
    def __init__(self) -> None:
        super().__init__()
