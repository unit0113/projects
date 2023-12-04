import pygame


class UI:
    def __init__(self) -> None:
        self.text_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.large_text_font = pygame.font.SysFont("Consolas", 36)

    def draw_text(
        self,
        window: pygame.surface.Surface,
        text: str,
        font: pygame.font,
        text_color: str,
        x: int,
        y: int,
    ) -> None:
        img = font.render(text, True, text_color)
        window.blit(img, (x, y))
