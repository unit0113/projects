import pygame


class Crosshair:
    def __init__(self, img: pygame.surface.Surface) -> None:
        self.img = img
        self.rect = self.img.get_rect()

        # Hide mouse
        pygame.mouse.set_visible(False)

    def draw(self, window: pygame.surface.Surface) -> None:
        self.rect.center = pygame.mouse.get_pos()
        window.blit(self.img, self.rect)
