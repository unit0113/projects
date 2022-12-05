

def magnitude(data: list) -> int:
    left = data[0]
    if isinstance(left, list):
        left = magnitude(left)

    right = data[1]
    if isinstance(right, list):
        right = magnitude(right)

    return 3 * left + 2 * right


def add(left: list, right: list) -> list:
    result = [left] + [right]
    return result


def explode(data: list):
    path_to_explode = find_path_to_explode(data, [])

    if path_to_explode:
        sub_data = data
        for index in path_to_explode:
            sub_data = sub_data[index]
        
        left = sub_data[0]
        right = sub_data[1]

        # Add left
        if 1 in path_to_explode:
            last_1 = last_index(last_index, 1)




        # Add right
        if 0 in path_to_explode:
            last_0 = last_index(path_to_explode, 0)
            sub_data = data
            for index in path_to_explode[:last_0]:
                sub_data = sub_data[index]

            while isinstance(sub_data, list):
                sub_data = sub_data[0]

            sub_data += right

    print(data)



def find_path_to_explode(data: list, path: list) -> list:
    if path and len(path) >= 4:
        return path


    sub_data = data
    if path:
        for index in path:
            sub_data = sub_data[index]

    for index, elem in enumerate(sub_data):
        if isinstance(elem, list):
            new_path = path + [index]
            return find_path_to_explode(data, new_path)

    return []


def last_index(data: list, val: int):
    rev_data = data[::-1]
    i = rev_data.index(val)
    return len(data) - i - 1


data = [[[[[9,8],1],2],3],4]
explode(data)
print(data)