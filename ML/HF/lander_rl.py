import gym
from stable_baselines3 import PPO, ppo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


# Create and reset environment
'''env = gym.make("LunarLander-v2")
observation = env.reset()'''

'''# Check what environment looks like
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())'''

'''
for _ in range(20):
    action = env.action_space.sample()
    print(f'Action taken: {action}')

    observation, reward, done, info = env.step(action)

    if done:
        print('Reseting')
        observation = env.reset()'''

# Creating vectorized environments
#env = make_vec_env('LunarLander-v2', n_envs=16)

env = gym.make("LunarLander-v2")
# Create agent
model = model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)
# Train agent
model.learn(total_timesteps=500000)
# Save model
model_name = r"NNs\HF\ppo-LunarLander-v2"
model.save(model_name)

# Evaluate model
eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")