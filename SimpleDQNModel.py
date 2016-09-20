import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class SimpleDQNModel(object):
  def __init__(self, env, hidden_1=16, hidden_2=16,
               initial_learning_rate=0.1, learning_rate_decay_factor=0.96,
               num_steps_per_decay=1000, gamma=0.99, tau=0.01,
               soft_updates=False):
    """Two layer neural network for Q-function approximation in
    Q-learning for OpenAI gym.

    arguments:
    env -- the OpenAI gym environment

    keyword arguments:
    hidden_1 -- no of neurons in the first hidden layer. default 16
    hidden_2 -- no of neurons in the second hidden layer. default 16
    initial_learning_rate -- for gradient descent. default 0.1
    learning_rate_decay_factor -- for gradient descent. default 0.96
    num_steps_per_decay -- decay schedule for gradient descent.
                           default 1000
    gamma -- Q-learning gamma. default 0.99
    tau -- Soft target update rate. default 0.01
    soft_updates -- soft target updates. default False
    """
    self.input_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
    self.hidden_1 = hidden_1
    self.hidden_2 = hidden_2
    self.gamma = gamma
    self.tau = tau
    self.soft_updates = False

    self.experience_record = []
    self.inputs = tf.placeholder(
        tf.float32, shape=[None, self.input_size], name='inputs')
    self.targets = tf.placeholder(
        tf.float32, shape=[None, self.action_size], name='targets')

    self.online_model = self.build_net()
    self.target_model = self.build_net()

    if self.soft_updates:
      self.target_updates = self.get_soft_updates()
    else:
      self.target_updates = self.get_hard_updates()

    self.online_outputs = self.online_model['outputs']
    self.target_outputs = self.target_model['outputs']
    self.loss = tf.reduce_mean(
        tf.pow(self.online_outputs - self.targets, 2))

    self.global_step = tf.Variable(0, trainable=False)
    self.lr = tf.train.exponential_decay(initial_learning_rate,
                                         self.global_step,
                                         num_steps_per_decay,
                                         learning_rate_decay_factor,
                                         staircase=True)
    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train = self.optimizer.minimize(
        self.loss, global_step=self.global_step)
    init = tf.initialize_all_variables()

    self.sess = tf.Session()
    self.sess.run(init)

  def build_net(self):
    """Builds a two layer neural network. Returns a dictionary
    containing weight variables and outputs.
    """
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
    q = self.sess.run(self.online_outputs, feed_dict={
      self.inputs:observation})
    return q

  def infer_target_q(self, observation):
    q = self.sess.run(self.target_outputs, feed_dict={
      self.inputs:observation})
    return q

  def get_hard_updates(self):
    """Builds operations for assigning online model weights to
    target model.
    """
    updates = []
    for key, value in self.online_model.items():
      if key is not 'outputs':
        W = self.target_model.get(key)
        update = W.assign(value)
        updates.append(update)

    return updates

  def get_soft_updates(self):
    """Builds operations for assigning online model weights
    incrementally to target model.
    """
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
    """Perform one step of gradient descent training on batch.
    Also update target model if soft updates are enabled.
    """
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

    if self.soft_updates:
      self.do_target_updates()

  def post_episode(self):
    """Perform once per episode tasks here. Currently only hard updates.
    """
    if not self.soft_updates:
      self.do_target_updates()
