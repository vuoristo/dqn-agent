""" Parent class for DQNModels. Defines the interface for communication
with DQNAgent.
"""
import tensorflow as tf
import numpy as np

class DQNModel(object):
  def __init__(
      self, env, learning_rate=2.5e-4, momentum=0.95, gamma=0.99, tau=0.01,
      clip_delta=1.0, soft_updates=True, steps_to_hard_update=10000,
      weights_to_load=None, train_dir='train', collect_summaries=True):
    """
    arguments:
    env -- OpenAI gym environment

    keyword arguments:
    learning_rate -- RMSProp learning rate
    momentum -- RMSProp momentum
    gamma -- Q-learning gamma. default 0.99
    tau -- Soft target update rate. default 0.01
    clip_delta -- Limit abs(delta) to clip_delta value. default 1.0
    soft_updates -- soft target updates. default True
    steps_to_hard_update -- number of steps between hard updates.
                            default 10000
    weights_to_load -- Path to pretrained weights. default None
    train_dir -- directory where summaries and variables are saved.
                 default train
    collect_summaries -- toggle summary collection. default True
    """

    self.num_actions = env.action_space.n
    self.gamma = gamma
    self.tau = tau
    self.soft_updates = soft_updates
    self.steps_to_hard_update = steps_to_hard_update
    self.total_steps = 0
    self.collect_summaries = collect_summaries
    self.clip_delta = clip_delta

    with tf.variable_scope('inputs'):
      self.first_observation = tf.placeholder(
          tf.float32, shape=[None] + list(self.input_shape),
          name='first_observation')
      self.second_observation = tf.placeholder(
          tf.float32, shape=[None] + list(self.input_shape),
          name='second_observation')
      self.actions = tf.placeholder(
          tf.int32, shape=[None, 1], name='actions')
      self.rewards = tf.placeholder(
          tf.float32, shape=[None, 1], name='rewards')
      self.terminals_mask = tf.placeholder(
          tf.float32, shape=[None, 1], name='terminals_mask')

    with tf.variable_scope('online_model'):
      self.online_model = self.build_net(self.first_observation, trainable=True)
    with tf.variable_scope('target_model'):
      self.target_model = self.build_net(self.second_observation, trainable=False)

    online_qs = self.online_model['outputs']
    target_qs = self.target_model['outputs']

    with tf.variable_scope('train_targets'):
      actions_mask = tf.one_hot(self.actions, self.num_actions,
                                name='actions_mask')
      actions_mask = tf.reshape(actions_mask, (-1, self.num_actions))
      masked_online_qs = tf.reduce_sum(actions_mask*online_qs,
          reduction_indices=1)
      masked_online_qs = tf.reshape(masked_online_qs, (-1, 1))
      max_q_t_1 = tf.reduce_max(target_qs, reduction_indices=1)
      max_q_t_1 = tf.reshape(max_q_t_1, (-1, 1))

      self.delta = self.terminals_mask * (
          gamma * max_q_t_1 + self.rewards
          ) - masked_online_qs

      if self.clip_delta:
        tf.clip_by_value(self.delta, -self.clip_delta, self.clip_delta,
            name='clipped_delta')

    with tf.variable_scope('main_loss'):
      self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

    self.optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                               momentum=momentum,
                                               epsilon=0.01)
    self.train = self.optimizer.minimize(self.loss)

    # The target model is updated towards the online model in either
    # soft steps every iteration or by copying the weights all at once
    # every steps_to_hard_update steps
    with tf.variable_scope('target_updates'):
      if self.soft_updates:
        self.target_updates = self.get_soft_updates()
      else:
        self.target_updates = self.get_hard_updates()


    self.saver = tf.train.Saver(tf.all_variables())

    loss_summary = tf.scalar_summary('loss', self.loss)
    q_summary = tf.scalar_summary('max_q', tf.reduce_max(online_qs))
    self.summary_op = tf.merge_all_summaries()

    self.sess = tf.Session()
    self.summary_writer = tf.train.SummaryWriter(train_dir, self.sess.graph)
    if weights_to_load is None:
      init = tf.initialize_all_variables()
      self.sess.run(init)
    else:
      restore = self.saver.restore(self.sess, weights_to_load)

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

  def train_net(self, ob0, actions, rewards, ob1, terminals):
    """Perform one step of gradient descent training on batch.
    Also update target model if soft updates are enabled.
    """
    ob0s = self.reshape_input(ob0)
    ob1s = self.reshape_input(ob1)

    acs = np.reshape(actions, (-1, 1))
    res = np.reshape(rewards, (-1, 1))
    terms = np.reshape(terminals, (-1, 1))

    # Clip rewards to -1,0,1
    out_res = np.zeros_like(res)
    out_res[np.nonzero(res)] = 1. * np.sign(res[np.nonzero(res)])

    terminals_mask = np.invert(terms) * 1

    loss, _ = self.sess.run([self.loss, self.train], feed_dict={
        self.first_observation:ob0s,
        self.second_observation:ob1s,
        self.actions:acs,
        self.rewards:out_res,
        self.terminals_mask:terminals_mask})

    self.total_steps += 1

    if self.soft_updates:
      self.do_target_updates()
    else:
      if self.total_steps % self.steps_to_hard_update == 0:
        self.do_target_updates()

    if self.collect_summaries and self.total_steps % 100 == 0:
      summary_str = self.sess.run(self.summary_op, feed_dict={
        self.first_observation:ob0s,
        self.second_observation:ob1s,
        self.actions:acs,
        self.rewards:out_res,
        self.terminals_mask:terminals_mask})
      self.summary_writer.add_summary(summary_str, self.total_steps)

    if self.total_steps % 10000 == 0:
      self.saver.save(self.sess,
                      'train/model.ckpt',
                      global_step=self.total_steps)

  def get_q_value(self, observation):
    ob1s = self.reshape_input(observation)
    return self.infer_online_q(ob1s)
