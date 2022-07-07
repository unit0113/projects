from turtle import *


def tree(size, levels, angle):
    # Base case
    if levels == 0:
        color('green')
        dot(size)
        color('black')
        return

    # Draw stem
    forward(size)

    # Draw right branch
    right(angle)
    tree(size*0.8, levels-1, angle)

    # Draw left branch
    left(angle*2)
    tree(size*0.8, levels-1, angle)

    # Return turtle to starting point
    right(angle)
    backward(size)


def snowflake_side(length, levels):
    if levels == 0:
        forward(length)
        return

    length /= 3.0
    snowflake_side(length, levels-1)
    left(60)
    snowflake_side(length, levels-1)
    right(120)
    snowflake_side(length, levels-1)
    left(60)
    snowflake_side(length, levels-1)


def create_snowflake(sides, length):
    for _ in range(sides):
        snowflake_side(length, sides)
        right(360 / sides)


speed(0)
#left(90)
#tree(70, 8, 30)

create_snowflake(5, 500)

mainloop()