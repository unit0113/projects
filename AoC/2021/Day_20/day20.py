from copy import deepcopy


def get_inputs(path):
    with open(path, 'r') as file:
        image_enhancer = file.readline().strip()
        raw_image = [['1' if char == '#' else '0' for char in list(line.strip())] for line in file.readlines()[1:]]

    return image_enhancer, raw_image


def conv_to_int(x, y, image):
    result = ''
    for rel_x in range(x - 1, x + 2):
        result += ''.join(image[rel_x][y - 1: y + 2])

    return int(result, 2)


def pad(image):
    n = len(image[0]) + 4

    # Pad top
    image.insert(0, ['0'] * n)
    image.insert(0, ['0'] * n)

    # Pad mid
    for row in image[2:]:
        row.insert(0, '0')
        row.insert(0, '0')
        row.extend(['0', '0'])

    # Pad bottom
    image.extend([['0'] * n, ['0'] * n])


def enhance(image, image_enhancer):
    enhanced_image = deepcopy(image)
    stop = len(image[0]) - 1
    for x in range(1, stop):
        for y in range(1, stop):
            enhanced_image[x][y] = decode(x, y, image, image_enhancer)

    return enhanced_image


def decode(x, y, image, image_enhancer):
    return '1' if image_enhancer[conv_to_int(x, y, image)] == '#' else '0'


def enhance_cycle(image, image_enhancer, num_iters):
    for _ in range(num_iters):
        image = pad(image)
        image = enhance(image, image_enhancer)

    return image


def counter(image):
    return sum([row.count('1') for row in image])


if __name__ == '__main__':
    #image_enhancer, raw_image = get_inputs(r'AoC\2021\Day_20\test_input.txt')
    image_enhancer, raw_image = get_inputs(r'AoC\2021\Day_20\input.txt')
    enhanced_image = enhance_cycle(raw_image, image_enhancer, 2)
    print(f'Part 1: {counter(enhanced_image)}')

    #enhanced_image = enhance_cycle(raw_image, image_enhancer, 50)
    #print(f'Part 2: {counter(enhanced_image)}')
    