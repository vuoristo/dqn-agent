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

    self.input_shape = env.observation_space.shape
    self.num_actions = env.action_space.n
    self.gamma = gamma
    self.tau = tau
    self.soft_updates = soft_updates

    self.inputs = tf.placeholder(
        tf.float32, shape=[None] + list((self.input_shape)), name='inputs')
    self.targets = tf.placeholder(
        tf.float32, shape=[None, self.num_actions], name='targets')

    # Build two identical models, one for online inference
    # the other for training targets
    self.online_model = self.build_net(self.inputs)
    self.target_model = self.build_net(self.inputs)

    self.loss = tf.reduce_mean(
        tf.pow(self.online_model['outputs'] - self.targets, 2))

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
    ob0s, acs, res, ob1s = zip(*batch)
    ob0s = self.reshape_input(ob0s)
    ob1s = self.reshape_input(ob1s)
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

    if self.soft_updates:
      self.do_target_updates()

  def post_episode(self):
    """Perform once per episode tasks here. Currently only hard updates.
    """
    if not self.soft_updates:
      self.do_target_updates()
