import random


def main():
    level = get_level()
    score = 0
    for _ in range(10):
        num_wrong = 0
        num1 = generate_integer(level)
        num2 = generate_integer(level)
        answer = str(num1 + num2)
        guess = 0
        while num_wrong < 3:
            guess = input(f'{num1} + {num2} = ')
            if guess == answer:
                score += 1
                break
            else:
                num_wrong += 1
                if num_wrong == 3:
                    print(f'{num1} + {num2} = {answer}')
                else:
                    print('EEE')

    print(score)


def get_level():
    level = 0
    while not (0 < level < 4):
        try:
            level = int(input("Level: "))
        except ValueError:
            continue

    return level


def generate_integer(level):
    if level == 1:
        start = 0
    else:
        start = 10 ** (level-1)

    return random.randint(start, 10 ** (level)-1)


if __name__ == "__main__":
    main()
