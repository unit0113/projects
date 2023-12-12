from .pw_standard_cannon import StandardCannon

ENEMY_SHIP_DATA = {
    "bug_3": {
        "hp": 25,
        "sprite_sheet": "bug_3",
        "primary_weapons": [[StandardCannon, ((32, 0), 0, "RedThinLong")]],
        "secondary_weapons": [],
        "movement_behavior": "forward_behavior",
        "movement_behavior_args": [100],
        "fire_behavior": "random_single_fire_behavior",
        "fire_behavior_args": [500],
    },
    "bug_3_dt": {
        "hp": 25,
        "sprite_sheet": "bug_3",
        "primary_weapons": [[StandardCannon, ((32, 0), 0, "RedThinLong")]],
        "secondary_weapons": [],
        "movement_behavior": "s_behavior",
        "movement_behavior_args": [100],
        "fire_behavior": "random_double_tap_fire_behavior",
        "fire_behavior_args": [750],
    },
    "bug_3_b": {
        "hp": 25,
        "sprite_sheet": "bug_3",
        "primary_weapons": [[StandardCannon, ((32, 0), 0, "RedThinLong")]],
        "secondary_weapons": [],
        "movement_behavior": "stall_behavior",
        "movement_behavior_args": [100],
        "fire_behavior": "random_burst_fire_behavior",
        "fire_behavior_args": [1000, 4],
    },
}
