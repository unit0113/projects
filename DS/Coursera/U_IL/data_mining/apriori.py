import os
import sys
from collections import defaultdict
from itertools import combinations


MIN_SUPPORT = 0.01


def read_data() -> tuple[defaultdict[str, int], int]:
    path = os.path.join(sys.path[0], 'categories.txt')

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
    frequent_dict = get_frequent(data_dict, count)
    path = os.path.join(sys.path[0], 'patterns.txt')

    with open(path, 'w') as file:
        for catgegory, support in reversed(sorted(frequent_dict.items(), key=lambda kv: kv[1])):
            file.write(f"{support}:{catgegory}\n")


def part2(data_dict: defaultdict[str, int], count: int) -> None:
    frequent_dict = get_frequent(data_dict, count)
    read_path = os.path.join(sys.path[0], 'categories.txt')
    write_path = os.path.join(sys.path[0], 'patterns.txt')

    n = 2
    while frequent_dict:
        category_dict = defaultdict(lambda: 0)
        combos = get_combinations(frequent_dict, n)
        with open(read_path, 'r') as file:
            for line in file.readlines():
                line_list = line.strip().split(';')
                for item_set in combos:
                    if all(cat in line_list for cat in item_set):
                        category_dict[item_set] += 1
        n += 1
        frequent_dict = get_frequent(category_dict, count)
        with open(write_path, 'a') as file:
            for catgegories, support in reversed(sorted(frequent_dict.items(), key=lambda kv: kv[1])):
                file.write(f"{support}:{';'.join(catgegories)}\n")


def get_combinations(frequent_dict: defaultdict[str, int], length):
    if length == 2:
        return list(combinations(frequent_dict.keys(), 2))
    
    combos = []
    for index, set1_keys in enumerate(frequent_dict.keys()):
        for set2_keys in list(frequent_dict.keys())[index + 1:]:
            combo = tuple(sorted(set(set1_keys + set2_keys)))
            if len(combo) == length and combo not in combos:
                combos.append(combo)

    return combos


if __name__ == '__main__':
    data_dict, count = read_data()
    part1(data_dict, count)
    part2(data_dict, count)
