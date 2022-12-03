import regex as re


class Target:
    def __init__(self) -> None:
        with open(r'AoC\2021\Day_17\input.txt', 'r') as file:
            string_data = file.readline()
            matches = re.search(r"x=(-?\d+)..(-?\d+), y=(-?\d+)..(-?\d+)", string_data)
            self.x_min, self.x_max, self.y_min, self.y_max = [int(num) for num in matches.groups()]

    def is_valid_x(self, x: int) -> bool:
        return self.x_min <= x <= self.x_max

    def is_valid_y(self, y: int) -> bool:
        return self.y_min <= y <= self.y_max

    def is_valid(self, x: int, y: int) -> bool:
        return self.is_valid_x(x) and self.is_valid_y(y)


def part2(target: Target) -> int:
    
    def hits_target(vx, vy):
        x = y = 0
        while True:
            # breaking conditions
            if (x > target.x_max
                or vx == 0 and not target.x_min <= x <= target.x_max
                or vx == 0 and y < target.y_min):
                 return False
            
            # target condition
            if target.is_valid(x, y):
                return True
            
            x += vx
            y += vy
            
            vx = max(vx - 1, 0)
            vy -= 1
    

    distinct_velocitys = 0
    
    for vx in range(target.x_max + 1):
        for vy in range(target.y_min, -target.y_min + 1):
            distinct_velocitys += hits_target(vx, vy)
            
    return distinct_velocitys


def main():
    target = Target()

    # Part 1
    n = target.y_min * -1 -1
    print(n * (n + 1) / 2)

    # Part 2
    print(part2(target))


if __name__ == "__main__":
    main()