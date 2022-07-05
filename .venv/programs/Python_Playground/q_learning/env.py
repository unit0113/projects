import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import os


style.use('ggplot')

SIZE = 10
EPISODES = 25_000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
SHOW_EVERY = 3000

epsilon = 0.9
EPSILON_DECAY = 0.9998
LEARNING_RATE = 0.1
DISCOUNT = 0.95

start_q_table = None
PLAYER_COLOR = (255, 175, 0)
FOOD_COLOR = (0, 255, 0)
ENEMY_COLOR = (0, 0, 255)


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f'{self.x}, {self.y}'

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=1)

    def move(self, x=None, y=None):
        if not x:
            self.x += np.random.randint(-1, 2)
            self.y += np.random.randint(-1, 2)
        else:
            self.x += x
            self.y += y

        # Account for walls
        self.x = max(0, self.x)
        self.x = min(self.x, SIZE - 1)
        self.y = max(0, self.y)
        self.y = min(self.y, SIZE - 1)


if not start_q_table:
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, 'rb') as file:
        q_table = pickle.load(file)


episode_rewards = []
for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if not episode % SHOW_EVERY:
        print(f'Episode {episode} epsilon: {epsilon}')
        print(f'{SHOW_EVERY} episode mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True
    
    else:
        show = False

    episode_reward = 0
    for _ in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)
        enemy.move()

        if player == enemy:
            reward = -ENEMY_PENALTY
        elif player == food:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q


        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.x][food.y] = FOOD_COLOR
            env[player.x][player.y] = PLAYER_COLOR
            env[enemy.x][enemy.y] = ENEMY_COLOR
            img = Image.fromarray(env, 'RGB')
            img = img.resize((500, 500))
            cv2.imshow("image", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward')
plt.xlabel(f'Episode')
plt.show()

with open(f'Python_Playground\q_learning\qtable-{int(time.time())}.pickle', 'wb') as file:
    pickle.dump(q_table, file)


