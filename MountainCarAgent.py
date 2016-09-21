import gym

from SimpleDQNModel import SimpleDQNModel
from DQNAgent import DQNAgent

ENV_NAME = 'MountainCar-v0'
def main():
  env = gym.make(ENV_NAME)
  model = SimpleDQNModel(env, hidden_1=10, hidden_2=6, num_steps_per_decay=1000, soft_updates=True, gamma=0.99, tau=0.001, initial_learning_rate=0.01, learning_rate_decay_factor=0.999)
  agent = DQNAgent(env, model, epsilon=0.3, epsilon_decay=0.99, max_steps=5000)
  agent.train()

if __name__ == '__main__':
  main()
