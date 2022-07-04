import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import os

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25_000
END_EPSILON_DECAY = 5_000
SHOW_EVERY = 500

epsilon = 0.5
epsilon_decay_value = epsilon / END_EPSILON_DECAY

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}



def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def get_action(state):
    return np.argmax(q_table[get_discrete_state(state)])

for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2])
        else:
            action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        '''if not episode % SHOW_EVERY:
            env.render()'''

        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state + (action, )] = new_q

        if new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    
    epsilon = max(0, epsilon - epsilon_decay_value)
    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode}  avg: {average_reward}  min: {aggr_ep_rewards['min'][-1]}  max: {aggr_ep_rewards['max'][-1]}")

env.close()

filename = os.path.join(r'Python_Playground\q_learning', 'final-qtable.npy')
np.save(filename, q_table)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='Average')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='Minimum')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='Maximum')
plt.legend(loc=4)
plt.show()
