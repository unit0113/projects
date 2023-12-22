import random

from .enemy import Enemy


class EnemyFactory:
    @staticmethod
    def get_enemy(level: int, faction: str) -> Enemy:
        enemy = Enemy(f"{faction}_{random.randint(1, 6)}")

        start_position_data = enemy.get_valid_start_positions()
        direction = random.choice(list(start_position_data.keys()))
        x = random.randint(
            start_position_data[direction]["start_x"],
            start_position_data[direction]["end_x"],
        )
        y = random.randint(
            start_position_data[direction]["start_y"],
            start_position_data[direction]["end_y"],
        )

        enemy.set_start_position(x, y, 1, direction)
        return enemy
