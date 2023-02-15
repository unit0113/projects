import re
from itertools import combinations 
from collections import deque


with open(r'AoC\2021\Day_19\input.txt') as file:
    data = file.read()

with open(r'AoC\2021\Day_19\test_input.txt') as file:
    testdata = file.read()


def read_data(data):
    L = []
    scanners = (x.strip() for x in re.split(r'\s*--- scanner \d+ ---\s*', data) if x)
    for scanner in scanners:
        lines = scanner.split()
        rows = [tuple(int(x) for x in line.split(',')) for line in lines]
        L.append(rows)
    return L


def add(a, b):
    return tuple(aa + bb for (aa, bb) in zip(a, b))


def subtract(a, b):
    return tuple(aa - bb for (aa, bb) in zip(a, b))


def distance_sq(a, b):
    return sum(d**2 for d in subtract(a, b))


def manhattan(a, b):
    return sum(abs(d) for d in subtract(a, b))


def pairwise_distance(scanner):
    D = {}
    dupes = set()
    for a, b in combinations(scanner, 2):
        dist = distance_sq(a, b)
        if dist in D:
            dupes.add(dist)
        D[dist] = (a, b)
    for dupe in dupes:
        del D[dupe]
    return D


def sorted_diff_lookup(scanner):
    sortedrows = list(sorted(scanner))
    D = {}
    dupes = set()
    for a, b in zip(sortedrows[1:], sortedrows):
        diff = subtract(a, b)
        if diff in D:
            dupes.add(diff)
        else:
            D[diff] = (a, b)
    for dupe in dupes:
        del D[dupe]
    return D


def rotate_xy(scanner):
    for _ in range(4):
        yield scanner
        scanner = [(-y, x, z) for (x, y, z) in scanner]


def iter_orientations(scanner):
    yield from rotate_xy([(x, y, z) for (x, y, z) in scanner])
    yield from rotate_xy([(-x, y, -z) for (x, y, z) in scanner])
    yield from rotate_xy([(x, -z, y) for (x, y, z) in scanner])
    yield from rotate_xy([(x, z, -y) for (x, y, z) in scanner])
    yield from rotate_xy([(-z, y, x) for (x, y, z) in scanner])
    yield from rotate_xy([(z, y, -x) for (x, y, z) in scanner])
        

def find_common_beacons(s0, s1):
    pw0 = pairwise_distance(s0)
    pw1 = pairwise_distance(s1)
    common_keys = set(pw0).intersection(pw1)
    if len(common_keys) > 50:
        return (
            {beacon for (k, v) in pw0.items() if k in common_keys for beacon in v},
            {beacon for (k, v) in pw1.items() if k in common_keys for beacon in v}
        )

    
def align_axes(s0, s1):
    sdl0 = sorted_diff_lookup(s0)
    for i, sx in enumerate(iter_orientations(s1)):
        sdl1 = sorted_diff_lookup(sx)
        common_keys = set(sdl0).intersection(sdl1)
        if len(common_keys) > 6:
            break
    else:
        raise ValueError("No alignment")
    common_0 = {beacon for (k, v) in sdl0.items() if k in common_keys for beacon in v}
    common_1 = {beacon for (k, v) in sdl1.items() if k in common_keys for beacon in v}
    zeros = set(subtract(a, b) for a, b in zip(sorted(common_0), sorted(common_1)))
    if len(zeros) > 1:
        raise ValueError("Too many zeros: %s" % zeros)
    zero = zeros.pop()
    return zero, i


scanners = read_data(data)
q = deque(scanners)
s0 = set(q.popleft())
zeros = {(0, 0, 0)}
while q:
    s1 = q.popleft()
    common_beacons = find_common_beacons(s0, s1)
    if common_beacons is None:
        q.append(s1)
        continue
    s1_zero, orientation_index = align_axes(*common_beacons)
    zeros.add(s1_zero)
    s1_orientations = iter_orientations(s1)
    for _ in range(orientation_index + 1):
        sx = next(s1_orientations)
    s0.update(add(s1_zero, x) for x in sx)
    

part_1 = len(s0)
print('part_1 =', part_1)

part_2 = max(manhattan(a, b) for (a, b) in combinations(zeros, 2))
print('part_2 =', part_2)