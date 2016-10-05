import gym
import numpy as np
import random
from collections import deque
from ExperienceMemory import ExperienceMemory

class DQNAgent(object):
  def __init__(self, env, model, max_episodes=200000, max_steps=1000000,
               exp_buffer_size=10000, epsilon=0.9, linear_epsilon_decay=True,
               epsilon_decay_steps=1.e6, exponential_epsilon_decay=0.99,
               min_epsilon=0.01, batch_size=20, render=True, warmup_steps=5e4):
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
    linear_epsilon_decay -- enable linear decay. True: linear,
                            False: exponential. default True
    epsilon_decay_steps -- how many steps for the epsilon to decay to
                           minimum
    exponential_epsilon_decay -- exponential decay factor for epsilon.
                                 default 0.99
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
    self.linear_epsilon_decay = linear_epsilon_decay
    self.epsilon_decay_steps = epsilon_decay_steps
    self.exponential_epsilon_decay = exponential_epsilon_decay

    self.step_log = deque(100*[0], 100)
    self.experiences = ExperienceMemory(exp_buffer_size)
    self.window_size = model.window_size
    self.model = model
    self.env = env

    self.warmup_steps = warmup_steps
    self.warmup = True

    self.render = render

  def train(self):
    """Steps environment and runs model training."""
    total_steps = 0
    for ep in range(self.max_episodes):
      ob0 = self.env.reset()
      step = 0
      while step < self.max_steps:
        action = self.select_action()
        ob1, reward, done, info = self.env.step(action)
        reward = reward if not done else 0
        self.save_and_train(ob0, action, reward, done, total_steps)
        if self.eps > self.min_epsilon:
          if self.linear_epsilon_decay:
            self.eps -= (1. - self.min_epsilon) / self.epsilon_decay_steps
          else:
            self.eps *= self.exponential_epsilon_decay
        if self.render:
          self.env.render()
        if done:
          self.report(step, ep)
          break
        if total_steps > self.warmup_steps:
          self.warmup = False

        ob0 = ob1
        step += 1
        total_steps += 1

  def select_action(self):
    """Selects action for given observation."""
    if self.warmup or \
        np.random.uniform(0,1) < self.eps:
      action = np.random.randint(self.n_actions)
    else:
      obs = self.experiences.get_current_window(self.window_size)
      q = self.model.get_q_value(obs)
      action = np.argmax(q)

    return action

  def save_and_train(self, ob0, ac, re, done, total_steps):
    """Saves experience, samples a batch of past experiences and runs
    one training step of the model.
    """
    ob0 = self.model.reshape_observation(ob0)
    self.experiences.save_experience(ob0, ac, re, done)

    if total_steps > self.batch_size:
      mb_ob0, mb_ac, mb_re, mb_ob1 = self.experiences.sample_minibatch(
          self.batch_size, self.window_size)
      self.model.train_net(mb_ob0, mb_ac, mb_re, mb_ob1)

  def report(self, steps, ep):
    self.step_log.append(steps)
    print('Episode: {} steps: {}, mean-100: {} epsilon: {}'.format(
      ep, steps, np.mean(self.step_log), self.eps))
