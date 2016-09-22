import gym

from SimpleDQNModel import SimpleDQNModel
from DQNAgent import DQNAgent

ENV_NAME = 'MountainCar-v0'
def main():
  env = gym.make(ENV_NAME)
  while 1:
    model = SimpleDQNModel(env, hidden_1=16, hidden_2=16, num_steps_per_decay=10000, soft_updates=True, gamma=0.999, tau=0.001, initial_learning_rate=0.01, learning_rate_decay_factor=0.95)
    agent = DQNAgent(env, model, epsilon=0.1, min_epsilon=0.001, epsilon_decay=0.99, max_steps=2000, batch_size=50)
    agent.train()

if __name__ == '__main__':
  main()
