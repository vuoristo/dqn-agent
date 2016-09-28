import gym
import numpy as np
import random
from collections import deque

EXPERIENCE_COLS = ['ob0', 'ac', 're', 'ob1', 'done']

class DQNAgent(object):
  def __init__(self, env, model, max_episodes=100000, max_steps=1000000,
               exp_buffer_size=10000, epsilon=0.9, epsilon_decay=0.99,
               min_epsilon=0.01, batch_size=20, render=True):
    """Deep Q-learning agent for OpenAI gym. Currently supports only
    one dimensional input.

    arguments:
    env -- the OpenAI gym environment
    model -- model for Q-function approximation

    keyword arguments:
    max_episodes -- default 100000
    max_steps -- max number of steps per episode. default 1000000
    exp_buffer_size -- how many experiences to remember. default 40000
    epsilon -- initial probability to take random action. default 0.99
    epsilon_decay -- exponential decay factor for epsilon. default 0.99
    min_epsilon -- minimum epsilon value. default 0.01
    batch_size -- number of elements in minibatch. default 20
    render -- enable environment rendering every timestep. default True
    """
    self.n_actions = env.action_space.n
    self.max_episodes = max_episodes
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.exp_buffer_size = exp_buffer_size
    self.eps = epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay = epsilon_decay

    self.step_log = deque(100*[0], 100)
    self.experiences = []
    self.window_size = model.window_size
    self.model = model
    self.env = env

    self.render = render

  def train(self):
    """Steps environment and runs model training."""
    for ep in range(self.max_episodes):
      ob0 = self.env.reset()
      step = 0
      while step < self.max_steps:
        action = self.select_action()
        ob1, reward, done, info = self.env.step(action)
        reward = reward if not done else 0
        self.save_and_train(ob0, action, reward, ob1, done)
        if self.render:
          self.env.render()
        if done:
          self.report(step, ep)
          self.model.post_episode()
          if self.eps > self.min_epsilon:
            self.eps *= self.epsilon_decay
          break

        ob0 = ob1
        step += 1

  def get_exp_window(self, index):
    end = index if index > 0 else 1
    start = end - self.window_size
    start = start if start >= 0 else 0
    sub_batch = self.experiences[start:end]
    while len(sub_batch) < self.window_size:
      sub_batch = [sub_batch[0]] + sub_batch

    window = {k: [] for k in EXPERIENCE_COLS}
    for exp in sub_batch:
      for key, value in exp.items():
        window[key] += [value]

    return window

  def select_action(self):
    """Selects action for given observation."""
    if len(self.experiences) == 0 or \
        np.random.uniform(0,1) < self.eps:
      action = np.random.randint(self.n_actions)
    else:
      exp = self.get_exp_window(len(self.experiences))
      q = self.model.get_q_value(exp)
      action = np.argmax(q)

    return action

  def sample_random_consecutive_batches(self):
    ends = np.random.randint(
        len(self.experiences) + 1, size=self.batch_size)
    # always use the latest experience
    ends = np.append(ends, len(self.experiences))

    batch = {k: [] for k in EXPERIENCE_COLS}
    for index in ends:
      sub_batch = self.get_exp_window(index)
      for key, value in sub_batch.items():
        batch[key] += value

    return batch

  def save_and_train(self, ob0, ac, re, ob1, done):
    """Saves experience, samples a batch of past experiences and runs
    one training step of the model.
    """
    ob0 = self.model.reshape_observation(ob0)
    ob1 = self.model.reshape_observation(ob1)
    experience = {
      'ob0':ob0,
      'ac':ac,
      're':re,
      'ob1':ob1,
      'done':done,
    }
    self.experiences.append(experience)
    mini_batch = self.sample_random_consecutive_batches()

    if len(self.experiences) > self.exp_buffer_size:
      self.experiences.pop(0)

    self.model.train_net(mini_batch)

  def report(self, steps, ep):
    self.step_log.append(steps)
    print('Episode: {} steps: {}, mean-100: {}'.format(
      ep, steps, np.mean(self.step_log)))
