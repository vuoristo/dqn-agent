""" Parent class for DQNModels. Defines the interface for
communication with DQNAgent.
"""
import tensorflow as tf
import numpy as np

class DQNModel(object):
  def __init__(
      self, env, initial_learning_rate=0.001,
      learning_rate_decay_factor=0.96, num_steps_per_decay=1000,
      gamma=0.99, tau=0.01, soft_updates=True):
    """
    arguments:
    env -- OpenAI gym environment

    keyword arguments:
    initial_learning_rate -- for gradient descent. default 0.001
    learning_rate_decay_factor -- for gradient descent. default 0.96
    num_steps_per_decay -- decay schedule for gradient descent.
                           default 1000
    gamma -- Q-learning gamma. default 0.99
    tau -- Soft target update rate. default 0.01
    soft_updates -- soft target updates. default True
    """

    self.num_actions = env.action_space.n
    self.gamma = gamma
    self.tau = tau
    self.soft_updates = soft_updates

    self.input_ob0s = tf.placeholder(
        tf.float32, shape=[None] + list(self.input_shape), name='ob0s')
    self.input_ob1s = tf.placeholder(
        tf.float32, shape=[None] + list(self.input_shape), name='ob1s')
    self.actions = tf.placeholder(
        tf.int32, shape=[None, self.window_size], name='action')
    self.rewards = tf.placeholder(
        tf.float32, shape=[None, self.window_size], name='rewards')
    self.rewards_mask = tf.placeholder(
        tf.float32, shape=[None, self.window_size], name='rewards_mask')

    self.online_model = self.build_net(self.input_ob0s)
    self.target_model = self.build_net(self.input_ob1s)

    online_qs = self.online_model['outputs']
    target_qs = self.target_model['outputs']
    actions_mask = tf.one_hot(self.actions, self.num_actions,
                              name='actions_mask')

    masked_online_qs = online_qs - actions_mask * online_qs
    train_targets = self.rewards_mask * (
        gamma * actions_mask * target_qs + self.rewards
        ) + masked_online_qs

    self.loss = tf.reduce_mean(
        tf.pow(online_qs - train_targets, 2))

    self.global_step = tf.Variable(0, trainable=False)
    self.lr = tf.train.exponential_decay(initial_learning_rate,
                                         self.global_step,
                                         num_steps_per_decay,
                                         learning_rate_decay_factor,
                                         staircase=True)
    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train = self.optimizer.minimize(
        self.loss, global_step=self.global_step)

    # The target model is updated towards the online model
    # in either soft steps every iteration or larger steps
    # every C iterations
    if self.soft_updates:
      self.target_updates = self.get_soft_updates()
    else:
      self.target_updates = self.get_hard_updates()

    init = tf.initialize_all_variables()

    self.sess = tf.Session()
    self.sess.run(init)

  def get_hard_updates(self):
    """Builds operations for assigning online model weights to
    target model.
    """
    updates = []
    for key, value in self.online_model['shared_vars'].items():
      W = self.target_model['shared_vars'].get(key)
      update = W.assign(value)
      updates.append(update)

    return updates

  def get_soft_updates(self):
    """Builds operations for assigning online model weights
    incrementally to target model.
    """
    updates = []
    for key, value in self.online_model['shared_vars'].items():
      W = self.target_model['shared_vars'].get(key)
      update = W.assign(self.tau * value + (1. - self.tau) * W)
      updates.append(update)

    return updates

  def do_target_updates(self):
    self.sess.run(self.target_updates)

  def train_net(self, batch):
    """Perform one step of gradient descent training on batch.
    Also update target model if soft updates are enabled.
    """
    ob0s = self.reshape_input(batch['ob0'])
    ob1s = self.reshape_input(batch['ob1'])
    acs = np.reshape(batch['ac'], [-1, self.window_size])
    res = np.reshape(batch['re'], [-1, self.window_size])
    rewards_mask = np.zeros_like(res)
    rewards_mask[np.nonzero(res)] = 1.

    loss, _ = self.sess.run([self.loss, self.train], feed_dict={
        self.input_ob0s:ob0s,
        self.input_ob1s:ob1s,
        self.actions:acs,
        self.rewards:res,
        self.rewards_mask:rewards_mask})

    if self.soft_updates:
      self.do_target_updates()

  def get_q_value(self, experience):
    ob1s = self.reshape_input(experience['ob1'])
    return self.infer_online_q(ob1s)

  def post_episode(self):
    """Perform once per episode tasks here. Currently only hard updates.
    """
    if not self.soft_updates:
      self.do_target_updates()
