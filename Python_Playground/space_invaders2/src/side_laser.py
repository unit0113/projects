from .laser_cannon import LaserCannon


class SideLaser:
    def __init__(
        self,
        offsets: list[tuple[int, int]],
        cooldown: float,
        projectile_color: str,
        base_damages: tuple[float, float],
        muzzle_velocity: int,
        dispersion: float,
    ) -> None:
        self.base_damages = base_damages
        self.base_cooldown = cooldown
        self.offsets = offsets
        self.cannons = [
            LaserCannon(
                offset,
                cooldown,
                f"{projectile_color}ThinLong",
                base_damages,
                muzzle_velocity,
                dispersion,
            )
            for offset in offsets
        ]

    def fire(self, ship_pos: tuple[int, int], direction: tuple[int, int]):
        return [cannon.fire(ship_pos, direction) for cannon in self.cannons]
