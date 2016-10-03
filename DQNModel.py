""" Parent class for DQNModels. Defines the interface for communication
with DQNAgent.
"""
import tensorflow as tf
import numpy as np

class DQNModel(object):
  def __init__(
      self, env, learning_rate=2.5e-4, momentum=0.95, gamma=0.99, tau=0.01,
      soft_updates=True, steps_to_hard_update=10000, train_dir='train'):
    """
    arguments:
    env -- OpenAI gym environment

    keyword arguments:
    learning_rate -- RMSProp learning rate
    momentum -- RMSProp momentum
    gamma -- Q-learning gamma. default 0.99
    tau -- Soft target update rate. default 0.01
    soft_updates -- soft target updates. default True
    steps_to_hard_update -- number of steps between hard updates.
                            default 10000
    """

    self.num_actions = env.action_space.n
    self.gamma = gamma
    self.tau = tau
    self.soft_updates = soft_updates
    self.steps_to_hard_update = steps_to_hard_update
    self.total_steps = 0

    self.input_ob0s = tf.placeholder(
        tf.float32, shape=[None] + list(self.input_shape),
        name='inputs_ob0s')
    self.input_ob1s = tf.placeholder(
        tf.float32, shape=[None] + list(self.input_shape),
        name='inputs_ob1s')
    self.actions = tf.placeholder(
        tf.int32, shape=[None, 1], name='actions')
    self.rewards = tf.placeholder(
        tf.float32, shape=[None, 1], name='rewards')
    self.rewards_mask = tf.placeholder(
        tf.float32, shape=[None, 1], name='rewards_mask')

    self.online_model = self.build_net(self.input_ob0s)
    self.target_model = self.build_net(self.input_ob1s)

    online_qs = self.online_model['outputs']
    target_qs = self.target_model['outputs']
    actions_mask = tf.one_hot(self.actions, self.num_actions,
                              name='actions_mask')

    # masked_online_qs is used to make the gradients of unselected
    # actions zero. The tensor contains the online network outputs
    # for actions not performed, making their effect on the loss zero.
    masked_online_qs = online_qs - actions_mask * online_qs

    # train_targets computes the target function for training the
    # action value function approximation. rewards_mask is a vector
    # containing zero for terminal observations and one for
    # non-terminals for making the targets of terminal actions zero.
    train_targets = self.rewards_mask * (
        gamma * actions_mask * target_qs + self.rewards
        ) + masked_online_qs

    self.loss = tf.reduce_mean(
        tf.pow(online_qs - train_targets, 2))

    self.optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                               momentum=momentum)
    self.train = self.optimizer.minimize(self.loss)

    # The target model is updated towards the online model in either
    # soft steps every iteration or by copying the weights all at once
    # every steps_to_hard_update steps
    if self.soft_updates:
      self.target_updates = self.get_soft_updates()
    else:
      self.target_updates = self.get_hard_updates()

    init = tf.initialize_all_variables()

    self.saver = tf.train.Saver(tf.all_variables())

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

    # Select the last actions and rewards of the window because only
    # the last actions and rewards of batch are used for training.
    acs = np.reshape(batch['ac'], [-1, self.window_size])
    res = np.reshape(batch['re'], [-1, self.window_size])
    acs = np.reshape(acs[:,-1], [-1,1])
    res = np.reshape(res[:,-1], [-1,1])

    # Clip rewards to -1,0,1
    out_res = np.zeros_like(res)
    out_res[np.nonzero(res)] = 1. * np.sign(res[np.nonzero(res)])

    rewards_mask = np.zeros_like(res)
    rewards_mask[np.nonzero(res)] = 1.

    loss, _ = self.sess.run([self.loss, self.train], feed_dict={
        self.input_ob0s:ob0s,
        self.input_ob1s:ob1s,
        self.actions:acs,
        self.rewards:out_res,
        self.rewards_mask:rewards_mask})

    self.total_steps += 1

    if self.soft_updates:
      self.do_target_updates()
    else:
      if self.total_steps % self.steps_to_hard_update == 0:
        self.do_target_updates()

    if self.total_steps % 10000 == 0:
      self.saver.save(self.sess,
                      'train/model.ckpt',
                      global_step=self.total_steps)

  def get_q_value(self, experience):
    ob1s = self.reshape_input(experience['ob1'])
    return self.infer_online_q(ob1s)
