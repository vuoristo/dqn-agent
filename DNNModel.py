from DQNModel import DQNModel
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class DNNModel(DQNModel):
  def __init__(self, env, window_size=1, hidden_1=16,
      hidden_2=16, **kwargs):
    self.input_shape = list(env.observation_space.shape)
    self.input_shape[0] *= window_size
    self.hidden_1 = hidden_1
    self.hidden_2 = hidden_2
    self.window_size = window_size
    super(DNNModel, self).__init__(env, **kwargs)

  def build_net(self, x):
    """Builds a two layer neural network. Returns a dictionary
    containing weight variables and outputs.
    """

    W_fc1 = weight_variable([self.input_shape[0], self.hidden_1])
    b_fc1 = bias_variable([self.hidden_1])
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = weight_variable([self.hidden_1, self.hidden_2])
    b_fc2 = bias_variable([self.hidden_2])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_outputs = weight_variable([self.hidden_2, self.num_actions])
    b_outputs = bias_variable([self.num_actions])

    outputs = tf.matmul(h_fc2, W_outputs) + b_outputs
    net_dict = {
        'shared_vars':{
          'W_fc1':W_fc1,
          'b_fc1':b_fc1,
          'W_fc2':W_fc2,
          'b_fc2':b_fc2,
          'W_outputs':W_outputs,
          'b_outputs':b_outputs,
        },
        'outputs':outputs,
    }

    return net_dict

  def reshape_input(self, observation):
    return np.reshape(observation, [-1, self.input_shape[0]])

  def infer_online_q(self, observation):
    q = self.sess.run(self.online_model['outputs'], feed_dict={
      self.inputs:observation})
    return q

  def infer_target_q(self, observation):
    q = self.sess.run(self.target_model['outputs'], feed_dict={
      self.inputs:observation})
    return q

  def reshape_observation(self, observation):
    return observation
