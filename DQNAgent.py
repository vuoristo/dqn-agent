import gym
import numpy as np
import random
from collections import deque

class DQNAgent(object):
  def __init__(self, env, model, max_episodes=100000, max_steps=1000000,
               exp_buffer_size=40000, epsilon=0.99, epsilon_decay=0.99,
               min_epsilon=0.01, batch_size=50):
    self.n_actions = env.action_space.n
    observation_shape = env.observation_space.shape
    assert np.ndim(observation_shape) == 1, 'Multidimensional input not supported'
    self.ob_size = observation_shape[0]
    self.max_episodes = max_episodes
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.exp_buffer_size = exp_buffer_size
    self.eps = epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay = epsilon_decay

    self.step_log = deque(100*[0], 100)
    self.experiences = []
    self.model = model
    self.env = env

  def train(self):
    for ep in range(self.max_episodes):
      ob0 = self.env.reset()
      step = 0
      while step < self.max_steps:
        action = self.act(ob0)
        ob1, reward, done, info = self.env.step(action)
        reward = reward if not done else 0
        self.save_and_train(ob0, action, reward, ob1)
        if done:
          self.report(step, ep)
          self.model.post_episode()
          if self.eps > self.min_epsilon:
            self.eps *= self.epsilon_decay
          break

        ob0 = ob1
        step += 1

  def act(self, observation):
    observation = np.reshape(observation, [-1,self.ob_size])
    if np.random.uniform(0,1) < self.eps:
      action = np.random.randint(self.n_actions)
    else:
      q = self.model.infer_online_q(observation)
      action = np.argmax(q)

    return action

  def save_and_train(self, ob0, ac, re, ob1):
    self.experiences.append((ob0, ac, re, ob1))
    if len(self.experiences) < self.batch_size:
      batch_size = len(self.experiences)
    else:
      batch_size = self.batch_size

    mini_batch = random.sample(self.experiences, batch_size)
    mini_batch.append(self.experiences[-1])

    if len(self.experiences) > self.exp_buffer_size:
      self.experiences.pop(0)

    self.model.train_net(mini_batch)

  def report(self, steps, ep):
    self.step_log.append(steps)
    print('Episode: {} steps: {}, mean-100: {}'.format(ep, steps, np.mean(self.step_log)))
