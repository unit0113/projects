from .laser_cannon import LaserCannon


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
                    muzzle_offset, 250, f"{projectile_color}ThinLong", 25, 1000, 2
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset, 500, f"{projectile_color}ThinLong", 50, 1500, 0
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset, 400, f"{projectile_color}ThickLong", 50, 750, 1
                )
            elif name == "gatling":
                return LaserCannon(
                    muzzle_offset, 50, f"{projectile_color}ThinShort", 5, 1000, 1
                )
        else:
            if name == "standard":
                return LaserCannon(
                    muzzle_offset, 0, f"{projectile_color}ThinLong", 25, 1000, 2
                )
            elif name == "sniper":
                return LaserCannon(
                    muzzle_offset, 0, f"{projectile_color}ThinLong", 50, 1500, 0
                )
            elif name == "turbolaser":
                return LaserCannon(
                    muzzle_offset, 0, f"{projectile_color}ThickLong", 50, 750, 1
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
                return [
                    LaserCannon(offset, 250, f"{projectile_color}ThinLong", 10, 1000, 2)
                    for offset in muzzle_offsets
                ]
