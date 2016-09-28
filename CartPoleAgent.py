import gym

from DNNModel import DNNModel
from DQNAgent import DQNAgent

ENV_NAME = 'CartPole-v0'
def main():
  env = gym.make(ENV_NAME)
  model = DNNModel(env, gamma=0.99, hidden_1=256, hidden_2=16,
                   soft_updates=False, window_size=4)
  agent = DQNAgent(env, model, epsilon=1, linear_epsilon_decay=True,
                   epsilon_decay_steps=1.e6, exp_buffer_size=500000,
                   batch_size=50, render=False)
  agent.train()

if __name__ == '__main__':
  main()
