from particle import Particle
import colors
import random
import pygame


class Sparkler:
    def __init__(self):
        self.sparks = []

    def create_sparks(self, x, y):
        self.sparks.append(Particle(x, y, random.randint(-10, 10), random.randint(-10, 10), 3, colors.PURPLE))

    def update(self, surface):
        surface.fill(colors.BLACK)
        for spark in self.sparks[:]:
            spark.update()
            spark.draw(surface)
            if spark.is_decayed:
                self.sparks.remove(spark)

        pygame.display.update()