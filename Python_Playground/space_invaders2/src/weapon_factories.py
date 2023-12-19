from .laser_cannon import LaserCannon
from .beam_weapon import BeamWeapon
from .side_laser import SideLaser
from .missile_launcher import MissileLauncher


class PrimaryWeaponFactory:
    @staticmethod
    def get_weapon(
        name: str,
        muzzle_offset: tuple[int, int],
        projectile_color: str,
        is_player: bool = False,
    ) -> LaserCannon:
        if is_player:
            if name == "standard":
                return LaserCannon(
                    muzzle_offset,
                    250,
                    f"{projectile_color}ThinLong",
                    (25, 25),
                    1000,
                    2,
                    (0, -1),
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset,
                    500,
                    f"{projectile_color}ThinLong",
                    (50, 50),
                    1500,
                    0,
                    (0, -1),
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset,
                    400,
                    f"{projectile_color}ThickLong",
                    (50, 50),
                    750,
                    1,
                    (0, -1),
                )
            elif name == "gatling":
                return LaserCannon(
                    muzzle_offset,
                    50,
                    f"{projectile_color}ThinShort",
                    (5, 5),
                    1000,
                    3,
                    (0, -1),
                )
            elif name == "beam":
                return BeamWeapon(
                    muzzle_offset,
                    2500,
                    500,
                    projectile_color,
                    (1, 1),
                    6,
                    (0, -1),
                )
        else:
            if name == "standard":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    f"{projectile_color}ThinLong",
                    (25, 25),
                    1000,
                    2,
                    (0, 1),
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    f"{projectile_color}ThinLong",
                    (50, 50),
                    1500,
                    0,
                    (0, 1),
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    f"{projectile_color}ThickLong",
                    (50, 50),
                    750,
                    1,
                    (0, 1),
                )


class SecondaryWeaponFactory:
    @staticmethod
    def get_weapon(
        name: str,
        muzzle_offsets: tuple[tuple[int, int]],
        projectile_color: str,
        is_player: bool = False,
    ) -> LaserCannon:
        if is_player:
            if name == "side_cannon":
                return SideLaser(
                    muzzle_offsets, 250, projectile_color, (10, 10), 1000, 2, (0, -1)
                )
            elif name == "missile":
                return MissileLauncher(
                    muzzle_offsets, 1000, (0, 100), 600, "missile", (0, -1)
                )

        else:
            if name == "side_cannon":
                return SideLaser(
                    muzzle_offsets, 0, projectile_color, (10, 10), 1000, 2, (0, 1)
                )
            elif name == "missile":
                return MissileLauncher(
                    muzzle_offsets, 1000, (0, 100), 600, "missile", (0, 1)
                )
