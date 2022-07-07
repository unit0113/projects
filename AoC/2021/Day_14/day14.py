from collections import defaultdict


insertions = {}
with open(r'AoC\2021\Day_14\test_input.txt', 'r') as file:
    template = file.readline().strip()
    file.readline
    for line in file.readlines()[1:]:
        line = line.strip().split(' -> ')
        insertions[line[0]] = line[1]


def polymerize(template, insertions, steps):
    for i in range(steps):
        result = ""
        template_list = list(template)

        for index, char in enumerate(template_list[:-1]):
            code = char + template_list[index+1]
            insertion = insertions[code]
            result += char + insertion

        result += template_list[-1]        
        template = result

    return result



def max_min_diff(template):
    template_list = list(template)
    counts = defaultdict(int)

    for char in template_list:
        counts[char] += 1

    max_count = max(counts.values())
    min_count = min(counts.values())

    return max_count - min_count


final_template = polymerize(template, insertions, 10)
difference = max_min_diff(final_template)
print(difference)