import pygame
from colors import BLACK, PURPLE, SILVER
from projectile import Projectile

class Show_Manager:
    def __init__(self, surface):
        self.surface = surface
        self.projectiles = []
        self.particles = []
        self.projectiles.append(Projectile(1700, 1400, 'high', PURPLE, 'enormous', PURPLE, SILVER))


    def update(self):
        particles_to_delete = []
        for particle in self.particles:
            particle.update()
            if particle.is_decayed:
                particles_to_delete.append(particle)

        for particle in particles_to_delete:
            self.particles.remove(particle)

        projectiles_to_delete = []
        for projectile in self.projectiles:
            trail = projectile.update()
            if trail:
                self.particles.append(trail)
            if projectile.timeout:
                self.particles.extend(projectile.burst())
                projectiles_to_delete.append(projectile)

        for projectile in projectiles_to_delete:
            self.projectiles.remove(projectile)

    def draw(self):
        self.surface.fill(BLACK)

        for particle in self.particles:
            particle.draw(self.surface)

        for projectile in self.projectiles:
            projectile.draw(self.surface)

        pygame.display.update()