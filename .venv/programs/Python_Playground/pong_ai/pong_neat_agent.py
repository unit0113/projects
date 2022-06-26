from curses.ascii import isdigit
from pong_neat import PongGame
import neat
import os
import pickle
import time
import pygame
import re

NUM_GENERATIONS = 50


class PongGameManager:
    def __init__(self):
        self.game = PongGame()

    def test_ai(self, net):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            self.game.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((self.game.right_paddle.rect.y,
                                   abs(self.game.right_paddle.rect.x - self.game.ball.rect.x),
                                   self.game.ball.rect.y))
            decision = output.index(max(output))

            if decision == 1:  # Move up
                self.game.paddle_up(left=False)
            else:  # Move down
                self.game.paddle_down(left=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.game.player_up()
            elif keys[pygame.K_DOWN]:
                self.game.player_down()

            self.game.draw()

    def train_ai(self, genome1, genome2, config):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play against eachother to determine their fitness.
        """
        start_time = time.time()

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        self.genome1 = genome1
        self.genome2 = genome2

        max_hits = 50

        while True:
            game_info = self.game.update()
            self.move_ai_paddles(net1, net2)
            self.game.draw()

            duration = time.time() - start_time
            if game_info.left_score + game_info.right_score == 3 or game_info.left_hits >= max_hits:
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, net1, net2):
        """
        Determine where to move the left and the right paddle based on the two 
        neural networks that control them. 
        """
        players = [(self.genome1, net1, self.game.left_paddle, True), (self.genome2, net2, self.game.right_paddle, False)]
        for (genome, net, paddle, left) in players:
            output = net.activate(
                (paddle.rect.y, abs(paddle.rect.x - self.game.ball.rect.x), self.game.ball.rect.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:  # Don't move
                genome.fitness -= 0.01  # we want to discourage this
            elif decision == 1:  # Move up
                valid = self.game.paddle_up(left=left)
            else:  # Move down
                valid = self.game.paddle_down(left=left)

            if not valid:  # If the movement makes the paddle go off the screen punish the AI
                genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration


def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i+1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            pong = PongGameManager()

            if pong.train_ai(genome1, genome2, config):
                quit()


def run_neat(config, menu_selection):
    # Change directory to save checkpoints in correct folder
    os.chdir(r'Python_Playground\pong_ai\neat_checkpoints')

    if menu_selection == 1:
        p = neat.Population(config)
        last_checkpoint_number = 0
    elif menu_selection == 2:
        checkpoints = os.listdir()
        if len(checkpoints) == 0:
            print('WARNING: No checkpoints found. Initializing new population.')
            p = neat.Population(config)
            last_checkpoint_number = 0
        else:
            checkpoints.sort(key=lambda x: os.stat(os.path.join(os.getcwd(), x)).st_ctime)
            last_checkpoint = checkpoints[-1]
            print(f'Loading {last_checkpoint}...')
            last_checkpoint_number = int(re.findall(r'\d+', last_checkpoint)[0])
            p = neat.Checkpointer.restore_checkpoint(last_checkpoint)
            print('Load successful. Resuming training.')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, NUM_GENERATIONS - last_checkpoint_number)
    os.chdir('..')
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    print("Training complete.")


def test_best_network(config):
    try:
        with open(r'Python_Playground\pong_ai\best.pickle', "rb") as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print('No best network exists. Please train a network')
        print('----------------------------------------------')
        main()
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    pong = PongGameManager()
    pong.test_ai(winner_net)


def menu():
    print('Welcome to Pong with NEAT AI!')
    print('-----------------------------')
    print('Options:')
    print('1. Train AI from scratch')
    print('2. Continue previous training')
    print('3. Play best AI')
    print('4. Quit')
    print()

    response = input('Select an option: ')
    while not response.isdigit() or int(response) < 1 or int(response) > 4:
        response = input('Invalid selection. Select an option (1-4): ')

    if response == '4':
        print('Goodbye!')
        quit()

    return int(response)


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    response = menu()
    if response == 1 or response == 2:
        run_neat(config, response)
    
    else:
        test_best_network(config)

if __name__ == '__main__':
    main()