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
        direction: tuple[int, int],
    ) -> None:
        self.base_damages = base_damages
        self.base_cooldown = cooldown
        self.offsets = offsets
        self.direction = direction
        self.cannons = [
            LaserCannon(
                offset,
                cooldown,
                f"{projectile_color}ThinLong",
                base_damages,
                muzzle_velocity,
                dispersion,
                self.direction,
            )
            for offset in offsets
        ]

    def fire(self, ship_pos: tuple[int, int]):
        return [cannon.fire(ship_pos) for cannon in self.cannons]

    def get_status(self) -> float:
        return self.cannons[0].get_status()
