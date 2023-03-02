import os
import sys
from collections import defaultdict


MIN_SUPPORT = 0.01


def read_data() -> tuple[defaultdict[str, int], int]:
    file_name = 'categories.txt'
    path = os.path.join(sys.path[0], file_name)

    category_dict = defaultdict(lambda: 0)
    category_count = 0
    with open(path, 'r') as file:
        for line in file.readlines():
            category_count += 1
            line_list = line.strip().split(';')
            for cat in line_list:
                category_dict[cat] += 1

    return category_dict, category_count


def get_frequent(data_dict: defaultdict[str, int], count: int) -> defaultdict[str, int]:
    support_dict = defaultdict(lambda: 0)
    req_support = int(count * MIN_SUPPORT)
    
    for key, value in data_dict.items():
        if value > req_support:
            support_dict[key] = value

    return support_dict


def part1(data_dict: defaultdict[str, int], count: int) -> None:
    support_dict = get_frequent(data_dict, count)
    print_to_file(support_dict, 'patterns.txt')


def print_to_file(data_dict: dict[str, int], file_name: str) -> None:
    path = os.path.join(sys.path[0], file_name)

    with open(path, 'w') as file:
        for catgegory, support in reversed(sorted(data_dict.items(), key=lambda kv: kv[1])):
            file.write(f"{support}:{catgegory}\n")






if __name__ == '__main__':
    data_dict, count = read_data()
    part1(data_dict, count)