import random


limit = 0
while limit < 1:
    try:
        limit = int(input("Level: "))
    except ValueError:
        continue

secret_num = random.randint(1, limit)
guess = 0
while guess != secret_num:
    try:
        guess = int(input("Guess: "))
    except ValueError:
        continue
    if guess < 1:
        continue

    if guess < secret_num:
        print("Too small!")
    elif guess > secret_num:
        print("Too large!")

print("Just right!")
