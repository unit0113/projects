import pygame

from .ship import Ship
from .enemy_ship_data import ENEMY_SHIP_DATA
from .settings import (
    FRAME_TIME,
    ENEMY_BASE_HP,
    ENEMY_BASE_SHIELD_COOLDOWN,
    ENEMY_BASE_SHIELD_REGEN,
    ENEMY_BASE_SHIELD_STRENGTH,
    ENEMY_BASE_SPEED,
)

from .behavior_forward import ForwardBehavior
from .behavior_stall import StallBehavior
from .behavior_s import SBehavior
from .behavior_zig_zag import ZigZagBehavior
from .behavior_circle import CircleBehavior
from .behavior_side_circle import SideCircleBehavior
from .fire_behavior_laser import LaserFireBehavior
from .fire_behavior_beam import BeamFireBehavior

BEHAVIORS = {
    "forward_behavior": ForwardBehavior,
    "stall_behavior": StallBehavior,
    "s_behavior": SBehavior,
    "zig_zag_behavior": ZigZagBehavior,
    "circle_behavior": CircleBehavior,
    "side_circle_behavior": SideCircleBehavior,
    "laser_fire_behavior": LaserFireBehavior,
    "beam_fire_behavior": BeamFireBehavior,
}


class Enemy(Ship, pygame.sprite.Sprite):
    def __init__(self, ship_type: str, sprite_sheet: pygame.Surface) -> None:
        Ship.__init__(self)
        pygame.sprite.Sprite.__init__(self)

        # Set basic ship data
        self.ship_type = ship_type
        ship_data = ENEMY_SHIP_DATA[ship_type]
        self.health = ship_data["multipliers"]["hp"] * ENEMY_BASE_HP
        self.speed = ship_data["multipliers"]["speed"] * ENEMY_BASE_SPEED
        self.secondary_offsets = ship_data["secondary_offsets"]
        self.projectile_color = ship_data["projectile_color"]

        # Set shield baseline stats
        self.base_shield_strength = (
            ship_data["multipliers"]["shield_strength"] * ENEMY_BASE_SHIELD_STRENGTH
        )
        self.base_shield_cooldown = (
            ship_data["multipliers"]["shield_cooldown"] * ENEMY_BASE_SHIELD_COOLDOWN
        )
        self.base_shield_regen = (
            ship_data["multipliers"]["shield_regen"] * ENEMY_BASE_SHIELD_REGEN
        )

        self.sprites = self.split_sprite_sheet(sprite_sheet, 1, 6, scale=1)

        # Image and animation data
        self.orientation = "level"
        self.frame_index = 0
        self.image = self.sprites[self.orientation][self.frame_index]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
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

    def get_valid_start_positions(self) -> dict:
        """Returns the movement behavior's starting location data

        Returns:
            dict: valid start positions of the assigned movement behavior
        """
        return self.movement_behavior.valid_start_locations

    def get_group_data(self) -> dict:
        """Returns the movement behavior's group starting location data

        Returns:
            dict: valid start positions of the assigned movement behavior for groups
        """
        return self.movement_behavior.group_data

    def set_start_condition(
        self,
        x: int,
        y: int,
        behavior_jerk: float,
        direction: str,
        fire_behavior_multiplier: float,
        health_multiplier: float,
        add_shield: bool,
        shield_strength_multipler: float,
        speed_multiplier: float,
    ) -> None:
        """Initialize the position and capabilities of the ship based on the movement behavior and level difficulty

        Args:
            x (int): Starting X position
            y (int): Starting Y position
            behavior_jerk (float): Jerk to be assigned to movement behavior
            direction (str): Direction of the movement behavior
            fire_behavior_multiplier (float): Fire rate cooldown modifier
            health_multiplier (float): Health modifier
            add_shield (bool): Whether to add shield to the ship
            shield_strength_multipler (float): Shield strength modifier
            speed_multiplier (float): Speed modifier
        """
        # Update stats
        self.pos = pygame.Vector2(x, y)
        self.rect.center = self.pos
        self.movement_behavior.set_starting_values(behavior_jerk, direction)
        self.fire_behavior.set_level_improvement(fire_behavior_multiplier)
        self.health *= health_multiplier
        self.max_health = self.health
        self.speed *= speed_multiplier

        # Initialize ship point value
        self.points = (
            self.movement_behavior.get_points()
            + self.fire_behavior.get_points()
            * (len(self.primary_weapons) + len(self.secondary_weapons) / 2)
            * health_multiplier
            * speed_multiplier
        )

        if add_shield:
            self.base_shield_strength *= shield_strength_multipler
            self.add_shield(1.5)
            self.points *= shield_strength_multipler

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
        """Creates projectiles if the ship is able to fire

        Returns:
            Optional[Laser, Missile, Torpedo]: projectiles fired during this frame
        """
        projectiles = []
        if self.movement_behavior.can_fire() and self.fire_behavior.can_fire():
            for weapon in self.primary_weapons:
                projectile = weapon.fire(self.rect.bottomleft)
                if projectile:
                    projectiles.append(projectile)
            for weapon in self.secondary_weapons:
                projectile = weapon.fire(self.rect.topleft)
                if projectile:
                    projectiles.append(projectile)
            if projectiles:
                self.fire_behavior.fire()

        return projectiles

    def get_points(self) -> float:
        """Return the point value of the ship

        Returns:
            float: Ship's point value
        """
        return self.points

    def __str__(self) -> str:
        return f"{self.ship_type}: ({self.pos[0]}, {self.pos[1]}); {type(self.movement_behavior)}"
