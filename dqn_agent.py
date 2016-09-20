import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class SimpleDQNModel(object):
  def __init__(self, env, hidden_1=16, hidden_2=16,
               initial_learning_rate=0.1, learning_rate_decay_factor=0.96,
               num_steps_per_decay=1000, gamma=0.99, tau=0.1):
    self.input_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
    self.hidden_1 = hidden_1
    self.hidden_2 = hidden_2
    self.gamma = gamma
    self.tau = tau

    self.experience_record = []
    self.inputs = tf.placeholder(tf.float32, shape=[None, self.input_size], name='inputs')
    self.targets = tf.placeholder(tf.float32, shape=[None, self.action_size], name='targets')

    self.online_model = self.build_net()
    self.target_model = self.build_net()

    self.target_updates = self.get_soft_updates()

    self.online_outputs = self.online_model['outputs']
    self.target_outputs = self.target_model['outputs']
    self.loss = tf.reduce_mean(tf.pow(self.online_outputs - self.targets, 2))

    self.global_step = tf.Variable(0, trainable=False)
    self.lr = tf.train.exponential_decay(initial_learning_rate,
                                         self.global_step,
                                         num_steps_per_decay,
                                         learning_rate_decay_factor,
                                         staircase=True)
    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train = self.optimizer.minimize(self.loss, global_step=self.global_step)
    init = tf.initialize_all_variables()

    self.sess = tf.Session()
    self.sess.run(init)

  def build_net(self):
    W_fc1 = weight_variable([self.input_size, self.hidden_1])
    b_fc1 = bias_variable([self.hidden_1])
    h_fc1 = tf.nn.relu(tf.matmul(self.inputs, W_fc1) + b_fc1)

    W_fc2 = weight_variable([self.hidden_1, self.hidden_2])
    b_fc2 = bias_variable([self.hidden_2])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_outputs = weight_variable([self.hidden_2, self.action_size])
    b_outputs = bias_variable([self.action_size])
    outputs = tf.matmul(h_fc2, W_outputs) + b_outputs

    net_dict = {
        'W_fc1':W_fc1,
        'b_fc1':b_fc1,
        'W_fc2':W_fc2,
        'b_fc2':b_fc2,
        'W_outputs':W_outputs,
        'b_outputs':b_outputs,
        'outputs':outputs,
    }

    return net_dict

  def infer_online_q(self, observation):
    q = self.sess.run(self.online_outputs, feed_dict={self.inputs:observation})
    return q

  def infer_target_q(self, observation):
    q = self.sess.run(self.target_outputs, feed_dict={self.inputs:observation})
    return q

  def get_soft_updates(self):
    updates = []
    for key, value in self.online_model.items():
      if key is not 'outputs':
        W = self.target_model.get(key)
        update = W.assign(self.tau * value + (1. - self.tau) * W)
        updates.append(update)

    return updates

  def do_target_updates(self):
    self.sess.run(self.target_updates)

  def train_net(self, batch):
    ob0s, acs, res, ob1s = zip(*batch)
    ob0s = np.reshape(ob0s, [-1,self.input_size])
    ob1s = np.reshape(ob1s, [-1,self.input_size])
    acs = np.reshape(acs, [-1])
    res = np.reshape(res, [-1])

    # hack for zeroing gradients from Keras-rl
    tgt_qs = self.infer_online_q(ob0s)

    q_t1s = np.max(self.infer_target_q(ob1s), axis=1)

    for tgt, ac, re, q_t1 in zip(tgt_qs, acs, res, q_t1s):
      if re == 0:
        tgt[ac] = 0
      else:
        tgt[ac] = re + self.gamma * q_t1

    loss, _ = self.sess.run([self.loss, self.train], feed_dict={
        self.inputs:ob0s,
        self.targets:tgt_qs})

    self.do_target_updates()

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

ENV_NAME = 'CartPole-v0'
def main():
  env = gym.make(ENV_NAME)
  model = SimpleDQNModel(env)
  agent = DQNAgent(env, model)
  agent.train()

if __name__ == '__main__':
  main()
