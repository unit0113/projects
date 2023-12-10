from .pw_standard_cannon import StandardCannon

PLAYER_SHIP_DATA = {
    "Vulcan": {
        "speed": 400,
        "hp": 100,
        "sprite_sheet": "vulcanA",
        "primary_weapons": [[StandardCannon, ((48, 0), 250, "GreenThinLong")]],
        "secondary_weapons": [],
    },
    "falcon": {"speed": 600, "hp": 50},
}
