import random
from scipy.stats import truncnorm

from .enemy import Enemy
from .settings import MIN_JERK, MAX_JERK, LEVEL_IMPROVEMENT_FACTOR


class EnemyFactory:
    def __init__(self, level: int, faction: str) -> None:
        self.improvement_dist = self._get_distribution(
            MIN_JERK,
            MAX_JERK,
            min(MAX_JERK, LEVEL_IMPROVEMENT_FACTOR ** (level - 1)),
            0.25,
        )
        self.faction = faction

    def _get_distribution(self, minimum: float, maximum: float, avg: float, std: float):
        dist = truncnorm(
            (minimum - avg) / std, (maximum - avg) / std, loc=avg, scale=std
        )
        return dist

    def get_enemy(self) -> Enemy:
        enemy = Enemy(f"{self.faction}_{random.randint(1, 6)}")

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

        factors = self.improvement_dist.rvs(6)

        enemy.set_start_condition(
            x,
            y,
            factors[0],
            direction,
            factors[1],
            factors[2],
            factors[3] > 1.5,
            factors[4],
            factors[5],
        )

        return enemy
