import numpy as np
from collections import deque
from pong import PongGame
import random
import torch
from pong_model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class Agent:

    def __init__(self):
        self.num_rounds = 0
        self.epsilon = 0.75
        self.gamma = 0.5
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(4, 256, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = [
            # Ball info
            game.ball.x_velocity,
            game.ball.y_velocity,

            # Self to ball comparision
            game.right_paddle.rect.x - game.ball.rect.x - game.ball.size,
            game.right_paddle.rect.y + game.right_paddle.length // 2 - game.ball.rect.y + game.ball.size // 2
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)        # Why zip?
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        if random.uniform(0,1) < self.epsilon:
            final_move[random.randint(0,2)] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            final_move[torch.argmax(prediction).item()] = 1

        return final_move


def train():
    agent = Agent()
    game = PongGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.initialize_round()
            agent.num_rounds += 1
            agent.train_long_memory()
            agent.model.save()
            if reward > -100:
                agent.epsilon -= 0.001

            print(f'Round {agent.num_rounds}\tComputer Score: {game.score_left}\tAI Score: {game.score_right}\tReward: {reward}')


if __name__ == '__main__':
    train()