import unittest
import numpy as np
import tensorflow as tf

from ConvModel import ConvModel
import gym

class ModelInputTestCase(unittest.TestCase):
  def tearDown(self):
    tf.reset_default_graph()

  def test_reshape_input(self):
    env = gym.make('Breakout-v0')
    model = ConvModel(env)
    test_input = [i * np.ones((84,84)) for i in range(4 * 8)]
    reshaped_obs = model.reshape_input(test_input)

    self.assertEqual(reshaped_obs.shape, (8,4,84,84))

    # check that the experience windows are ordered correctly
    for i in range(8):
      self.assertGreater(reshaped_obs[i,1,0,0], reshaped_obs[i,0,0,0])
      self.assertGreater(reshaped_obs[i,2,0,0], reshaped_obs[i,1,0,0])
      self.assertGreater(reshaped_obs[i,3,0,0], reshaped_obs[i,2,0,0])
      if i > 0:
        self.assertGreater(reshaped_obs[i,0,0,0], reshaped_obs[i-1,0,0,0])

  def test_model_shapes(self):
    env = gym.make('Breakout-v0')
    model = ConvModel(env)
    input_shape = [None, 4, 84, 84]
    self.assertEqual(model.first_observation.get_shape().as_list(),
        input_shape)
    self.assertEqual(model.second_observation.get_shape().as_list(),
        input_shape)

    self.assertEqual(model.actions.get_shape().as_list(), [None, 1])
    self.assertEqual(model.rewards.get_shape().as_list(), [None, 1])
    self.assertEqual(model.terminals_mask.get_shape().as_list(), [None, 1])

    o_qs = model.online_model['outputs']
    t_qs = model.online_model['outputs']
    self.assertEqual(o_qs.get_shape().as_list(), [None, 6])
    self.assertEqual(t_qs.get_shape().as_list(), [None, 6])

    self.assertEqual(model.train_targets.get_shape().as_list(), [None, 6])

    W_conv1 = model.online_model['shared_vars']['W_conv1']
    self.assertEqual(W_conv1.get_shape().as_list(), [8, 8, 4, 32])
