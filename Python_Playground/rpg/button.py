import pygame


class Button:
    def __init__(self, surface, x, y, image, size_x, size_y):
        self.image = pygame.transform.scale(image, (size_x, size_y))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False
        self.surface = surface

    def draw(self):
        action = False
        pos = pygame.mouse.get_pos()

        # check mouseover and clicked conditions
        if (
            self.clicked == False
            and pygame.mouse.get_pressed()[0] == 1
            and self.rect.collidepoint(pos)
        ):
            action = True
            self.clicked = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        # draw button
        self.surface.blit(self.image, (self.rect.x, self.rect.y))

        return action
