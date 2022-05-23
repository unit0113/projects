from turtle import Turtle, Screen
import random


NUM_RACERS = 8

COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'grey', 'turquoise', 'lime', 'magenta', 'dark goldenrod', 'dodger blue']

WIDTH = 1200
HEIGHT = 800

class Racer:
    def __init__(self, number, num_racers):
        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.shape('turtle')
        self.turtle.color(COLORS[number%12])
        self.number = number
        self.num_racers = num_racers

    
    def go_to_start(self):
        starting_y = 150 + (600 // self.num_racers) * self.number - HEIGHT // 2
        self.turtle.goto(-WIDTH // 2 + 20, starting_y)


    def forward(self):
        distance = random.randint(WIDTH // 600, WIDTH // 100)
        self.turtle.forward(distance)


def advance(racers):
    for racer in racers:
        racer.forward()


def create_racers(num_racers):
    racers = []
    for num in range(num_racers):
        racers.append(Racer(num, num_racers))

    for racer in racers:
        racer.go_to_start()

    return racers


def main():
    # Setup screen
    screen = Screen()
    screen.setup(WIDTH, HEIGHT)

    # Get user bet
    guess = screen.textinput(title="Place your bet", prompt="Who will win!? Nobody knows. Place your bet by entering a color: ").lower()

    # Create racers
    racers = create_racers(NUM_RACERS)

    screen.listen() 
    #screen.onkey(key='space', fun=advance(racers))

    racing = True
    while racing:
        advance(racers)
        for racer in racers:
            if racer.turtle.xcor() > WIDTH // 2 - 20:
                winning_color = racer.turtle.pencolor()
                if winning_color == guess:
                    print(f'Congradulations! The winner was {winning_color}')
                else:
                    print(f'Bad luck! The winner was {winning_color}')
                racing = False
                break

    screen.exitonclick()


if __name__ == "__main__":
    main()