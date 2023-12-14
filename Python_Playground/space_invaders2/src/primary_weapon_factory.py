from .pw_standard_cannon import StandardCannon


class PrimaryWeaponFactory:
    @staticmethod
    def get_weapon(
        name: str, muzzle_offset: tuple[int, int], is_player: bool = False
    ) -> StandardCannon:
        if is_player:
            if name == "standard":
                return StandardCannon(muzzle_offset, 250, "GreenThinLong", 25, 1000, 2)
            elif name == "sniper":
                return StandardCannon(muzzle_offset, 500, "BlueThinLong", 50, 1500, 0)
            elif name == "turbolaser":
                return StandardCannon(muzzle_offset, 400, "BlueThickLong", 50, 750, 1)
        else:
            if name == "standard":
                return StandardCannon(muzzle_offset, 0, "RedThinLong", 25, 1000, 2)
            elif name == "sniper":
                return StandardCannon(muzzle_offset, 0, "RedThinLong", 50, 1500, 0)
            elif name == "turbolaser":
                return StandardCannon(muzzle_offset, 0, "RedThickLong", 50, 750, 1)
