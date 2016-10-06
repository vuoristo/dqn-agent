import unittest
import numpy as np

from ConvModel import ConvModel
import gym

class ModelInputTestCase(unittest.TestCase):
  def setUp(self):
    self.env = gym.make('Breakout-v0')
    self.model = ConvModel(self.env)

  def test_reshape_input(self):
    test_input = [i * np.ones((84,84)) for i in range(4 * 8)]
    reshaped_obs = self.model.reshape_input(test_input)

    self.assertEqual(reshaped_obs.shape, (8,4,84,84))

    # check that the experience windows are ordered correctly
    for i in range(8):
      self.assertGreater(reshaped_obs[i,1,0,0], reshaped_obs[i,0,0,0])
      self.assertGreater(reshaped_obs[i,2,0,0], reshaped_obs[i,1,0,0])
      self.assertGreater(reshaped_obs[i,3,0,0], reshaped_obs[i,2,0,0])
      if i > 0:
        self.assertGreater(reshaped_obs[i,0,0,0], reshaped_obs[i-1,0,0,0])
