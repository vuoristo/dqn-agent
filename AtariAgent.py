import gym

from ConvModel import ConvModel
from DQNAgent import DQNAgent

ENV_NAME = 'Breakout-v0'
def main():
  env = gym.make(ENV_NAME)
  model = ConvModel(env, soft_updates=True)
  agent = DQNAgent(env, model)
  agent.train()

if __name__ == '__main__':
  main()
