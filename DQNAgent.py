import gym
import numpy as np
import random
from collections import deque
from ExperienceMemory import ExperienceMemory

class DQNAgent(object):
  def __init__(self, env, model, max_episodes=200000, max_steps=1000000,
               exp_buffer_size=40000, epsilon=0.9, linear_epsilon_decay=True,
               epsilon_decay_steps=1.e6, exponential_epsilon_decay=0.99,
               min_epsilon=0.01, batch_size=20, render=True, warmup_steps=5e4,
               update_freq=1):
    """Deep Q-learning agent for OpenAI gym. Currently supports only
    one dimensional input.

    arguments:
    env -- the OpenAI gym environment
    model -- model for Q-function approximation

    keyword arguments:
    max_episodes -- default 200000
    max_steps -- max number of steps per episode. default 1000000
    exp_buffer_size -- how many experiences to remember. default 40000
    epsilon -- initial probability to take random action. default 0.9
    linear_epsilon_decay -- enable linear decay. True: linear,
                            False: exponential. default True
    epsilon_decay_steps -- how many steps for the epsilon to decay to
                           minimum. default 1000000
    exponential_epsilon_decay -- exponential decay factor for epsilon.
                                 default 0.99
    min_epsilon -- minimum epsilon value. default 0.01
    batch_size -- number of elements in minibatch. default 20
    render -- enable environment rendering every timestep. default True
    warmup_steps -- how many steps to run before epsilon decay starts
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
    self.update_freq = update_freq

    self.reward_log = deque(100*[0], 100)
    self.window_size = model.window_size
    self.experiences = ExperienceMemory(exp_buffer_size)
    self.recent_observations = deque(maxlen=self.window_size)
    self.model = model
    self.env = env

    self.warmup_steps = warmup_steps
    self.warmup = True

    self.render = render

  def train(self):
    """ The training loop of the DQNAgent. Steps the environment,
    saves observations and calls model training.
    """
    total_steps = 0
    for ep in range(self.max_episodes):
      step = 0
      rewards = 0
      first_observation = self.env.reset()
      # recent_observations are episode specific
      self.recent_observations = deque(maxlen=self.window_size)
      self.append_to_recent_observations(first_observation)
      while step < self.max_steps:
        # select action according to the recent_observations
        action = self.select_action()
        second_observation, reward, done, _ = self.env.step(action)

        # observations are saved with the same index as the
        # action, reward and done following them
        self.save_experience(action, reward, done)
        self.append_to_recent_observations(second_observation)
        first_observation = second_observation

        rewards += reward
        step += 1
        total_steps += 1

        if self.render:
          self.env.render()
        if total_steps > self.warmup_steps:
          self.warmup = False
        if not self.warmup:
          if total_steps % self.update_freq == 0:
            self.train_model()
          if self.eps > self.min_epsilon:
            if self.linear_epsilon_decay:
              self.eps -= (1. - self.min_epsilon) / self.epsilon_decay_steps
            else:
              self.eps *= self.exponential_epsilon_decay
        if done:
          self.report(total_steps, step, rewards, ep)
          break

  def select_action(self):
    """Selects action for given observation."""
    if self.warmup or \
        np.random.uniform(0,1) < self.eps:
      action = np.random.randint(self.n_actions)
    else:
      obs = list(self.recent_observations)
      while len(obs) < self.window_size:
        obs = [obs[0]] + obs
      q = self.model.get_q_value(obs)
      action = np.argmax(q)

    return action

  def append_to_recent_observations(self, observation):
    observation = self.model.reshape_observation(observation)
    self.recent_observations.append(observation)

  def save_experience(self, action, reward, done):
    self.experiences.save_experience(self.recent_observations[-1],
        action, reward, done)

  def train_model(self):
    mb_ob0, mb_ac, mb_re, mb_ob1, mb_term = self.experiences.sample_minibatch(
        self.batch_size, self.window_size)
    self.model.train_net(mb_ob0, mb_ac, mb_re, mb_ob1, mb_term)

  def report(self, total_steps, steps, rewards, episode):
    self.reward_log.append(rewards)
    print('Episode: {} Total steps: {}, steps: {}, reward: {} mean-100: '
          '{} epsilon: {}'.format(episode, total_steps, steps, rewards,
          np.mean(self.reward_log), self.eps))
