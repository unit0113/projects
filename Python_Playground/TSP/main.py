from src.game import Game


def main():
    game = Game()

    while game.run:
        game.game_loop()


if __name__ == "__main__":
    main()
