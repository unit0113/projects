from .pw_standard_cannon import StandardCannon

PLAYER_SHIP_DATA = {
    "boomerang": {
        "speed": 400,
        "hp": 100,
        "sprite_sheet": "boomerang",
        "primary_weapons": [[StandardCannon, ((24, 0), 250, "GreenThinLong")]],
        "secondary_weapons": [],
    },
    "falcon": {"speed": 600, "hp": 50},
}
