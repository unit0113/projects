from .laser_cannon import LaserCannon
from .beam_weapon import BeamWeapon
from .side_laser import SideLaser
from .missile_launcher import MissileLauncher
from .torpedo_launcher import TorpedoLauncher


class PrimaryWeaponFactory(object):
    _instance = None
    _assets = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._assets = args[0]
            cls._instance = super(PrimaryWeaponFactory, cls).__new__(cls)
        return cls._instance

    def get_weapon(
        cls,
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
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (25, 25),
                    1000,
                    2,
                    (0, -1),
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset,
                    500,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (50, 50),
                    1500,
                    0,
                    (0, -1),
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset,
                    400,
                    cls._assets["projectiles"][f"laser{projectile_color}ThickLong"],
                    (50, 50),
                    750,
                    1,
                    (0, -1),
                )
            elif name == "gatling":
                return LaserCannon(
                    muzzle_offset,
                    50,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinShort"],
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
                    cls._assets["projectiles"][f"beam{projectile_color}"],
                    (2, 2),
                    6,
                    (0, -1),
                )
        else:
            if name == "standard":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (25, 25),
                    1000,
                    2,
                    (0, 1),
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (50, 50),
                    1500,
                    0,
                    (0, 1),
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset,
                    0,
                    cls._assets["projectiles"][f"laser{projectile_color}ThickLong"],
                    (50, 50),
                    750,
                    1,
                    (0, 1),
                )
            elif name == "beam":
                return BeamWeapon(
                    muzzle_offset,
                    0,
                    500,
                    cls._assets["projectiles"][f"beam{projectile_color}"],
                    (2, 2),
                    6,
                    (0, 1),
                )


class SecondaryWeaponFactory:
    _instance = None
    _assets = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._assets = args[0]
            cls._instance = super(SecondaryWeaponFactory, cls).__new__(cls)
        return cls._instance

    def get_weapon(
        cls,
        name: str,
        muzzle_offsets: tuple[tuple[int, int]],
        projectile_color: str,
        is_player: bool = False,
    ) -> LaserCannon:
        if is_player:
            if name == "side_cannon":
                return SideLaser(
                    muzzle_offsets,
                    250,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (10, 10),
                    1000,
                    2,
                    (0, -1),
                )
            elif name == "missile":
                return MissileLauncher(
                    muzzle_offsets,
                    1000,
                    (0, 50),
                    600,
                    (0, -1),
                    cls._assets["projectiles"]["missile"],
                )
            elif name == "torpedo":
                return TorpedoLauncher(
                    muzzle_offsets,
                    2000,
                    (0, 250),
                    400,
                    (0, -1),
                    cls._assets["projectiles"]["torpedo"],
                )

        else:
            if name == "side_cannon":
                return SideLaser(
                    muzzle_offsets,
                    0,
                    cls._assets["projectiles"][f"laser{projectile_color}ThinLong"],
                    (10, 10),
                    1000,
                    2,
                    (0, 1),
                )
            elif name == "missile":
                return MissileLauncher(
                    muzzle_offsets,
                    1000,
                    (0, 100),
                    600,
                    (0, 1),
                    cls._assets["projectiles"]["missile"],
                )
            elif name == "torpedo":
                return TorpedoLauncher(
                    muzzle_offsets,
                    0,
                    (0, 250),
                    400,
                    (0, 1),
                    cls._assets["projectiles"]["torpedo"],
                )
