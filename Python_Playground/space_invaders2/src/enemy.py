import pygame

from .ship import Ship
from .enemy_ship_data import ENEMY_SHIP_DATA
from .settings import FRAME_TIME

from .forward_behavior import ForwardBehavior
from .stall_behavior import StallBehavior
from .s_behavior import SBehavior
from .random_single_fire_behavior import RandomSingleFireBehavior
from .random_double_tap_fire_behavior import RandomDoubleTapFireBehavior
from .random_burst_fire_behavior import RandomBurstFireBehavior

from .shield import Shield

BEHAVIORS = {
    "forward_behavior": ForwardBehavior,
    "stall_behavior": StallBehavior,
    "s_behavior": SBehavior,
    "random_single_fire_behavior": RandomSingleFireBehavior,
    "random_double_tap_fire_behavior": RandomDoubleTapFireBehavior,
    "random_burst_fire_behavior": RandomBurstFireBehavior,
}


class Enemy(Ship, pygame.sprite.Sprite):
    def __init__(self, ship_type: str, x: int, y: int) -> None:
        Ship.__init__(self, ENEMY_SHIP_DATA[ship_type])
        pygame.sprite.Sprite.__init__(self)

        self.sprites = self.load_sprite_sheet(
            ENEMY_SHIP_DATA[ship_type]["sprite_sheet"], 1, 6
        )
        self.pos = pygame.Vector2(x, y)

        # Image and animation data
        self.orientation = "level"
        self.frame_index = 0
        self.image = self.sprites[self.orientation][self.frame_index]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.last_frame = pygame.time.get_ticks()

        # Weapon data
        self.load_weapons(ENEMY_SHIP_DATA[ship_type])

        # Behaviors
        self.movement_behavior = BEHAVIORS[
            ENEMY_SHIP_DATA[ship_type]["movement_behavior"]
        ](self.speed)
        self.fire_behavior = BEHAVIORS[ENEMY_SHIP_DATA[ship_type]["fire_behavior"]](
            *ENEMY_SHIP_DATA[ship_type]["fire_behavior_args"]
        )

        # Load shield
        if ENEMY_SHIP_DATA[ship_type]["shield_args"]:
            self.shield = Shield(
                *ENEMY_SHIP_DATA[ship_type]["shield_args"],
                self.image.get_width() // 1.5
            )

    def update(self, dt: float) -> None:
        """Update game object in game loop

        Args:
            dt (float): time since last frame
        """
        self.movement_behavior.update(dt)
        self.fire_behavior.update(dt)
        self.pos += self.movement_behavior.get_movement()
        self.rect.center = self.pos
        self.animate()
        if self.shield:
            self.shield.update(self.rect.center, self.last_hit)

    def animate(self) -> None:
        """Controls sprite animation of ship"""

        if pygame.time.get_ticks() > self.last_frame + FRAME_TIME:
            # Reset frame counter and increment frame
            self.last_frame = pygame.time.get_ticks()
            self.frame_index += 1
            if self.frame_index >= len(self.sprites[self.orientation]):
                # Loop animation
                self.frame_index = 0

            self.image = self.sprites[self.orientation][self.frame_index]

    def fire(self):
        projectiles = []
        if self.movement_behavior.can_fire() and self.fire_behavior.can_fire():
            for weapon in self.primary_weapons:
                projectile = weapon.fire(self.rect.bottomleft, (0, 1))
                if projectile:
                    self.fire_behavior.fire()
                    projectiles.append(projectile)
            for weapon in self.secondary_weapons:
                projectile = weapon.fire(self.rect.topleft, (0, 1))
                if projectile:
                    projectiles.append(projectile)

        return projectiles

    def draw(self, window: pygame.Surface) -> None:
        """Draws to the game window

        Args:
            window (pygame.Surface): pygame surface to draw on
        """

        window.blit(self.image, self.rect)
        if self.shield:
            self.shield.draw(window)
