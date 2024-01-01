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
        """Generate a single enemy

        Returns:
            Enemy: Single Enemy
        """
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

    def get_group(self) -> list[Enemy]:
        """Generate a coordinated group

        Returns:
            list[Enemy]: Group of enemies
        """
        enemy_type = random.randint(1, 6)
        enemy_template = Enemy(f"{self.faction}_{enemy_type}")

        group_data = enemy_template.get_group_data()
        spawn_timing = group_data["spawn_timing"]

        if spawn_timing == "simultaneous":
            return (
                self._get_simultaneous_group(enemy_type, group_data),
                spawn_timing,
            )
        elif spawn_timing == "sequential":
            return (
                self._get_sequential_group(
                    enemy_type, group_data, enemy_template.get_valid_start_positions()
                ),
                spawn_timing,
            )

    def _get_simultaneous_group(self, enemy_type: int, group_data: dict) -> list[Enemy]:
        """Generate a group of enemies that spawn simultaneously

        Args:
            enemy_type (int): Type of enemies to spawn
            group_data (dict): Grouping data of enemy

        Returns:
            list[Enemy]: Group of enemies
        """
        enemies = []

        # Select the slots to spawn in
        slots = random.sample(
            range(group_data["max_group_size"]),
            random.randint(2, group_data["max_group_size"]),
        )

        # Generate enemies
        for slot in slots:
            enemy = Enemy(f"{self.faction}_{enemy_type}")
            factors = self.improvement_dist.rvs(6)
            x = random.randint(
                group_data["starting_positions"][slot]["start_x"],
                group_data["starting_positions"][slot]["end_x"],
            )
            y = random.randint(
                group_data["starting_positions"][slot]["start_y"],
                group_data["starting_positions"][slot]["end_y"],
            )
            direction = group_data["starting_positions"][slot]["direction"]
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
            enemies.append(enemy)

        return enemies

    def _get_sequential_group(
        self, enemy_type: int, group_data: dict, start_position_data: dict
    ) -> list[Enemy]:
        """Generate a group of enemies that spawn sequentially

        Args:
            enemy_type (int): Type of enemies to spawn
            group_data (dict): Grouping data of enemy
            start_position_data (dict): Valid starting locations

        Returns:
            list[Enemy]: Group of enemies
        """
        enemies = []
        num_enemies = random.randint(2, group_data["max_group_size"])

        # Generate starting position
        direction = random.choice(list(start_position_data.keys()))
        x = random.randint(
            start_position_data[direction]["start_x"],
            start_position_data[direction]["end_x"],
        )
        y = random.randint(
            start_position_data[direction]["start_y"],
            start_position_data[direction]["end_y"],
        )

        # Generate enemies
        for _ in range(num_enemies):
            enemy = Enemy(f"{self.faction}_{enemy_type}")
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
            enemies.append(enemy)

        return enemies
