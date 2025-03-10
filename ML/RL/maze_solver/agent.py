from maze import Maze, make_gif
import pickle 
import random
import numpy as np
from PIL import Image


GAMMA = 0.9
alpha = 0.5

MAZE_SIZE = 8
EPISODES = 300
number_before_metrics = 10
GOAL = [2,0]
ACTIONS = ['UP','DOWN','RIGHT','LEFT']


def q(state, action, value_dict):
	result = value_dict.get(state, {}).get(action, 10)
	return result


def update(state, action, value, value_dict):
	state_values = value_dict.get(state, None)
	if state_values:
		state_values[action] = value
	else:
		value_dict[state] = {}
		value_dict[state][action] = value


def allowed_actions(input_state):
	state = input_state
	allowed = ['UP','DOWN','RIGHT','LEFT']
	last_line = state.split('\n')[-1]
	for c in last_line:
		if c == 'P':
			allowed.remove('DOWN')
	first_line = state.split('\n')[0]
	for c in first_line:
		if c == 'P':
			allowed.remove('UP')
	line_length = MAZE_SIZE+1
	for i in range(MAZE_SIZE):
		if state[line_length*i] == 'P':
			allowed.remove('LEFT')
		if state[line_length*i + line_length - 1] == 'P':
			allowed.remove('RIGHT')

	return allowed

def policy(values, state, epsilon = 0.1, verbose = False):
	best_action = None
	best_value = float('-inf')
	allowed = allowed_actions(state)
	random.shuffle(allowed)
	for action in allowed:
		if verbose:
			print(f'action: {action} value: {q(state, action, values)} vs best_value: {best_value}')
		if q(state, action, values) > best_value:
			best_value = q(state, action, values)
			if verbose:
				print(f'new best action: {action}')
			best_action = action

	r_var = random.random()
	if verbose:
		print(f'{r_var} vs {epsilon}')
	if r_var < epsilon:
		if verbose:
			print('choosing random')
		best_action = random.choice(allowed)
		best_value = q(state, best_action, values)
	if verbose:
		print(f'chose: {best_action}')
	return best_action, best_value

def state_values(values):
	state_matrix = [[0 for _ in range(MAZE_SIZE)] for _ in range(MAZE_SIZE)]

	for i in range(MAZE_SIZE):
		for j in range(MAZE_SIZE):
			mock_env = Maze(dimensions = [MAZE_SIZE, MAZE_SIZE])
			mock_env.position = np.asarray([i,j])
			mock_env.goal_position = np.asarray(GOAL)
			best_action = None
			state = mock_env.compressed_state_rep()
			best_value = float('-inf')
			allowed = allowed_actions(state)
			for action in allowed:
				if q(state, action, values) > best_value:
					best_value = q(state, action, values)
					best_action = action

			state_matrix[i][j] = round(best_value,1)

	for row in state_matrix:
		print(row)


def unique_starts(i):
	position = [0, 0]
	goal_position = GOAL

	return position, goal_position

def train(env, save_to = 'ML/RL/maze_solver/models/model3.dump'):
	env.reset()
	values = {}
	rewards_list = []
	steps_list = []
	step = 0
	last_path = []
	first_path = []
	
	for episoden in range(EPISODES):
		env.reset()
		position, goal_position = unique_starts(episoden)
		start_position = position
		env.position = np.asarray(position)
		env.goal_position = np.asarray(goal_position)
		current_state = env.compressed_state_rep()
		use_epsilon = 0.1
		if episoden > 200:
			use_epsilon = 0.0
		action, action_v = policy(values, current_state, epsilon = use_epsilon, verbose=(episoden==299))
		total_reward = 0
		#step = 0
		deltas = []

		while (not env.over):
			if episoden == 299:
				print(f'state: \n{current_state}')
				print(f'action chosen: {(action, action_v)}')
				last_path.append(action)
			if episoden == 0:
				first_path.append(action)

			step +=1
			reward = env.move(action)
			total_reward += reward
			next_state = env.compressed_state_rep()
			next_action, next_action_v = policy(values, next_state, epsilon = use_epsilon, verbose=(episoden==299))
			
			if env.over: #one can only win.
				next_action_v = 100
				total_reward+=100

			delta = next_action_v*GAMMA + reward - action_v
			deltas.append(delta)
			new_value = action_v + delta*alpha
			update(current_state, action, new_value, values)
			current_state = next_state
			action = next_action
			action_v = next_action_v

		rewards_list.append(total_reward)
		steps_list.append(step)

		if episoden > number_before_metrics:
			print(f'episode: {episoden} total_reward: {total_reward} agg: {sum(rewards_list[-number_before_metrics:])/number_before_metrics} ')
			print(f'original reward: {sum(rewards_list[:number_before_metrics])/number_before_metrics} ')
			print(f'explored states: {len(values.keys())} ')
			print(f'average delta: {sum(deltas)/len(deltas)}')
			print(f'steps: {step}')
			print(f'difference: {start_position} {goal_position}')
			state_values(values)

	print(values)
	with open('steps.txt', 'w') as f:
		output = str(steps_list)+'\n'+str(values)
		f.write(output)

	env.reset()
	position, goal_position = unique_starts(episoden)
	start_position = position
	env.position = np.asarray(position)
	env.goal_position = np.asarray(goal_position)
	boards = env.path(last_path)
	frames = [Image.fromarray(frame, 'RGB') for frame in boards]
	path = 'ML/RL/maze_solver/media/last_iter_5.gif'
	make_gif(frames, path)

	env.reset()
	position, goal_position = unique_starts(episoden)
	start_position = position
	env.position = np.asarray(position)
	env.goal_position = np.asarray(goal_position)
	boards = env.path(first_path)
	frames = [Image.fromarray(frame, 'RGB') for frame in boards]
	path = 'ML/RL/maze_solver/media/first_iter_5.gif'
	make_gif(frames, path)

	with open(save_to, 'wb') as handle:
		pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)

blocked_squares = [[i, i-1] for i in range(1,7)]
env = Maze(dimensions = [MAZE_SIZE, MAZE_SIZE], blocked = blocked_squares)
train(env)