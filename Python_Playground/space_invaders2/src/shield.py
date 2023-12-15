import pygame

SHIELD_BLUE_INNER = (63, 94, 249, 50)
SHIELD_BLUE_OUTER = (63, 94, 249, 200)


class Shield:
    def __init__(
        self, strength: int, regen: float, regen_cooldown: int, radius: int
    ) -> None:
        self.base_strength = strength
        self.strength = strength
        self.current_strength = strength
        self.base_regen = regen
        self.regen = regen
        self.base_regen_cooldown = regen_cooldown
        self.regen_cooldown = regen_cooldown
        self.level = 1
        self.last_hit = pygame.time.get_ticks()

        self.radius = radius
        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        self.image.fill((0, 0, 0))
        self.image.set_colorkey((0, 0, 0))
        self.image.set_alpha(100)
        self.rect = self.image.get_rect()

    @property
    def active(self) -> bool:
        return self.current_strength > 0

    def take_damage(self, damage: float) -> None:
        self.current_strength = max(0, self.current_strength - damage)
        self.last_hit = pygame.time.get_ticks()

    def update(self, pos: tuple[int, int]) -> None:
        self.rect.center = pos
        if (
            self.current_strength < self.strength
            and pygame.time.get_ticks() - self.last_hit > self.regen_cooldown
        ):
            self.current_strength = min(
                self.strength, self.current_strength + self.regen
            )

    def draw(self, window: pygame.Surface) -> None:
        self.image.set_alpha(100)
        if self.active:
            pygame.draw.circle(
                self.image, SHIELD_BLUE_INNER, (self.radius, self.radius), self.radius
            )
            window.blit(self.image, self.rect)
            pygame.draw.circle(
                self.image,
                SHIELD_BLUE_OUTER,
                (self.radius, self.radius),
                self.radius,
                max(1, int(2 * self.current_strength / self.base_strength)),
            )
            window.blit(self.image, self.rect)

    def upgrade(self) -> None:
        pass
