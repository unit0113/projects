import pygame

from .settings import SHIELD_BLUE_INNER, SHIELD_BLUE_OUTER


class Shield:
    def __init__(self, strength: int, regen: float, cooldown: int, radius: int) -> None:
        self.base_strength = strength
        self.max_strength = strength
        self.current_strength = strength
        self.base_regen = regen
        self.regen = regen
        self.base_cooldown = cooldown
        self.cooldown = cooldown
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
        """Whether shield is active

        Returns:
            bool: shield is active
        """

        return self.current_strength > 0

    def take_damage(self, damage: float) -> None:
        """Take damage to shields

        Args:
            damage (float): amount of damage
        """

        self.current_strength = max(0, self.current_strength - damage)
        self.last_hit = pygame.time.get_ticks()

    def update(self, pos: tuple[int, int], last_ship_hit: int) -> None:
        """Update game object in game loop

        Args:
            pos (tuple[int, int]): center position of hosting ship
            last_ship_hit (int): time in ms of last hit on hosting ship
        """

        self.rect.center = pos
        if (
            self.current_strength < self.max_strength
            and pygame.time.get_ticks() - max(self.last_hit, last_ship_hit)
            > self.cooldown
        ):
            self.current_strength = min(
                self.max_strength, self.current_strength + self.regen
            )

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        if self.active:
            self.image.set_alpha(100)
            pygame.draw.circle(
                self.image, SHIELD_BLUE_INNER, (self.radius, self.radius), self.radius
            )
            window.blit(self.image, self.rect)
            pygame.draw.circle(
                self.image,
                SHIELD_BLUE_OUTER,
                (self.radius, self.radius),
                self.radius,
                max(1, int(2 * self.current_strength / self.max_strength)),
            )
            window.blit(self.image, self.rect)

    def upgrade(self) -> None:
        pass

    def get_status(self) -> float:
        """Get shield status for UI

        Returns:
            float: shield strength as percent of max
        """
        return self.current_strength / self.max_strength
