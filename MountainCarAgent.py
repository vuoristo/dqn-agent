import gym

from DNNModel import DNNModel
from DQNAgent import DQNAgent

ENV_NAME = 'MountainCar-v0'
def main():
  env = gym.make(ENV_NAME)
  model = DNNModel(env, initial_learning_rate=0.01,
                   gamma=0.99, hidden_1=256, hidden_2=16,
                   soft_updates=True, window_size=4)
  agent = DQNAgent(env, model, epsilon=0.1,
                   epsilon_decay=0.97, min_epsilon=0.001,
                   exp_buffer_size=40000, batch_size=50,
                   max_steps=10000)
  agent.train()

if __name__ == '__main__':
  main()
