import gym
import argparse

from ConvModel import ConvModel
from DQNAgent import DQNAgent

ENV_NAME = 'Breakout-v0'
def main():
  parser = argparse.ArgumentParser('Train or Evaluate a DQN Agent for OpenAI '
      'Gym Atari Environments')
  parser.add_argument('--env', '-e', default=ENV_NAME)
  parser.add_argument('--evaluate', action='store_true', default=False)
  parser.add_argument('--load_weights', '-l', default=None)
  parser.add_argument('--render', '-r', action='store_true', default=False)

  args = parser.parse_args()
  env_name = args.env
  weights_to_load = args.load_weights
  evaluate = args.evaluate
  render = args.render

  env = gym.make(env_name)
  model = ConvModel(env, learning_rate=2.5e-4, momentum=0.95, gamma=0.99,
      tau=0.01, soft_updates=False, weights_to_load=weights_to_load)
  agent = DQNAgent(env, model, linear_epsilon_decay=True,
      epsilon_decay_steps=1.e6, epsilon=1.0, min_epsilon=0.1,
      exp_buffer_size=1000000, batch_size=32, render=render,
      update_freq=4, random_starts=30)

  if evaluate:
    agent.evaluate()
  else:
    agent.train()

if __name__ == '__main__':
  main()
