""" Defines a model using convolutional neural network for action value
function approximation using image inputs. The model takes window_size
consecutive images as the input. The consecutive images are considered
input features and are processed in one pass of the network.
"""
from DQNModel import DQNModel
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

def weight_variable(name, shape, trainable):
  return tf.get_variable(name,
      shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.001),
      trainable=trainable)

def bias_variable(name, shape, trainable):
  return tf.get_variable(name,
      shape=shape, initializer=tf.constant_initializer(value=0.01),
      trainable=trainable)

def conv2d(x,W,strides):
  return tf.nn.conv2d(x,W,strides=strides, padding='SAME')

class ConvModel(DQNModel):
  def __init__(self, env, resize_shape=(84, 84), crop_centering=(0.5,0.7),
              window_size=4, grayscale=True, **kwargs):
    """
    arguments:
    env -- OpenAI gym environment

    keyword arguments:
    resize_shape -- All input images are resized to this shape.
                    default (84,84)
    crop_centering -- Control the cropping position. Default (0.5,0.7)
    window_size -- Number of consecutive observations to feed to the
                   network. default 4
    grayscale -- Convert inputs to grayscale. default True
    """
    self.resize_shape = resize_shape
    self.crop_centering = crop_centering
    self.input_shape = [0,0,0]
    self.input_shape[1] = resize_shape[0]
    self.input_shape[2] = resize_shape[1]

    self.grayscale = grayscale
    if grayscale:
      self.input_shape[0] = window_size
    else:
      raise NotImplementedError()

    self.window_size = window_size
    super(ConvModel, self).__init__(env, **kwargs)

  def build_net(self, x, trainable=True):
    """Builds a convolutional neural network. Assumes square input
    images. Returns a dictionary containing weight variables and
    outputs.
    """

    x_input = tf.transpose(x, (0,2,3,1))

    with tf.variable_scope('conv1'):
      W_conv1 = weight_variable('W', [8, 8, self.input_shape[0], 32],
          trainable)
      b_conv1 = bias_variable('b', [32], trainable)

      h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1, [1,4,4,1]) +
          b_conv1, name='h')

    with tf.variable_scope('conv2'):
      W_conv2 = weight_variable('W', [4,4,32,64], trainable)
      b_conv2 = bias_variable('b', [64], trainable)

      h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1,2,2,1]) +
          b_conv2, name='h')

    with tf.variable_scope('conv3'):
      W_conv3 = weight_variable('W', [3,3,64,64], trainable)
      b_conv3 = bias_variable('b', [64], trainable)

      h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, [1,1,1,1]) +
          b_conv3, name='h')

    with tf.variable_scope('fc1'):
      conv3_out_size = 11*11*64
      W_fc1 = weight_variable('W', [conv3_out_size, 512], trainable)
      b_fc1 = bias_variable('b', [512], trainable)

      h_conv3_flat = tf.reshape(h_conv3, [-1, conv3_out_size])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name='h')

    with tf.variable_scope('output'):
      W_linear = weight_variable('W', [512, self.num_actions], trainable)
      b_linear = bias_variable('b', [self.num_actions], trainable)

      outputs = tf.matmul(h_fc1, W_linear) + b_linear

    net_dict = {
        'shared_vars':{
          'W_conv1': W_conv1,
          'b_conv1': b_conv1,
          'W_conv2': W_conv2,
          'b_conv2': b_conv2,
          'W_conv3': W_conv3,
          'b_conv3': b_conv3,
          'W_fc1': W_fc1,
          'b_fc1': b_fc1,
          'W_linear': W_linear,
          'b_linear': b_linear,
        },
        'outputs':outputs,
    }

    return net_dict

  def reshape_input(self, observation):
    return np.reshape(observation, (-1, self.input_shape[0],
      self.input_shape[1], self.input_shape[2]))

  def infer_online_q(self, observation):
    """ Run forward pass of the online net. input_ob0s is the input
    placeholder for the online_model.
    """
    q = self.sess.run(self.online_model['outputs'], feed_dict={
      self.first_observation:observation})
    return q

  def reshape_observation(self, observation):
    """ Crop non-square inputs to squares positioned with
    crop_centering. Resize to resize_shape. Optionally convert to
    grayscale.
    """
    img = Image.fromarray(observation)
    img = ImageOps.fit(img, self.resize_shape, centering=self.crop_centering)
    if self.grayscale:
      img = img.convert('L')
    return np.array(img)
