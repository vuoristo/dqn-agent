import gym

from ConvModel import ConvModel
from DQNAgent import DQNAgent

ENV_NAME = 'Breakout-v0'
def main():
  env = gym.make(ENV_NAME)
  model = ConvModel(env, learning_rate=2.5e-4, momentum=0.95, gamma=0.99,
      tau=0.01, soft_updates=False)
  agent = DQNAgent(env, model, linear_epsilon_decay=True,
      epsilon_decay_steps=1.e6, epsilon=1.0, min_epsilon=0.1,
      exp_buffer_size=1000000, batch_size=32, render=False,
      update_freq=4, random_starts=30)
  agent.train()

if __name__ == '__main__':
  main()
