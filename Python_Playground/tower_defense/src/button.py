import pygame


class Button:
    def __init__(
        self, pos: tuple[int, int], image: pygame.surface.Surface, single_click: bool
    ) -> None:
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos
        self.clicked = False
        self.single_click = single_click

    def draw(self, window: pygame.surface.Surface) -> bool:
        action = False
        # If mouseover
        if (
            not self.clicked
            and self.rect.collidepoint(pygame.mouse.get_pos())
            and pygame.mouse.get_pressed()[0] == 1
        ):
            action = True
            if self.single_click:
                self.clicked = True

        if not pygame.mouse.get_pressed()[0]:
            self.clicked = False

        window.blit(self.image, self.rect)
        return action
