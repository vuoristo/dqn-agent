import gym

from SimpleDQNModel import SimpleDQNModel
from DQNAgent import DQNAgent

ENV_NAME = 'CartPole-v0'
def main():
  env = gym.make(ENV_NAME)
  model = SimpleDQNModel(env, soft_updates=True)
  agent = DQNAgent(env, model)
  agent.train()

if __name__ == '__main__':
  main()
