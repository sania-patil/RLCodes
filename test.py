from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)


# from stable_baselines3 import PPO
# print("✅ Stable-Baselines3 is installed and working!")
