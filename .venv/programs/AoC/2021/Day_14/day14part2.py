from collections import defaultdict


insertions = {}
with open(r'AoC\2021\Day_14\input.txt', 'r') as file:
    template = file.readline().strip()
    file.readline
    for line in file.readlines()[1:]:
        line = line.strip().split(' -> ')
        insertions[line[0]] = line[1]


def make_pairs(template):
    results = defaultdict(int)
    template_list = list(template)
    for index, char in enumerate(template_list[:-1]):
        code = char + template_list[index+1]
        results[code] += 1

    return results


def polymerize_pairs(template, insertions, steps):
    for _ in range(steps):
        new_template = defaultdict(int)
        for pair, count in template.items():
            chars = list(pair)
            insertion = insertions[pair]
            pair1 = chars[0] + insertion
            pair2 = insertion + chars[1]
            new_template[pair1] += count
            new_template[pair2] += count
        
        template = new_template

    return template


def max_min_diff(template, org_template):
    counts = defaultdict(int)

    for pair, total in template.items():
        chars = list(pair)
        counts[chars[0]] += total
        counts[chars[1]] += total

    counts[org_template[0]] += 1
    counts[org_template[-1]] += 1

    max_count = max(counts.values())
    min_count = min(counts.values())

    return (max_count - min_count) // 2


paired_template = make_pairs(template)
final_template = polymerize_pairs(paired_template, insertions, 40)
difference = max_min_diff(final_template, template)
print(difference)