import gym

from DNNModel import DNNModel
from DQNAgent import DQNAgent

ENV_NAME = 'CartPole-v0'
def main():
  env = gym.make(ENV_NAME)
  model = DNNModel(env, initial_learning_rate=0.1,
                   gamma=0.99, hidden_1=10, hidden_2=6,
                   soft_updates=True)
  agent = DQNAgent(env, model, epsilon=0.9,
                   epsilon_decay=0.97,
                   exp_buffer_size=40000, batch_size=50)
  agent.train()

if __name__ == '__main__':
  main()
