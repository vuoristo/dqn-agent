import unittest

from ExperienceMemory import RingBuffer
from ExperienceMemory import ExperienceMemory

class RingBufferTestCase(unittest.TestCase):
  def test_fill_buffer_simple(self):
    buf_len = 100
    buf = RingBuffer(buf_len)
    for i in range(buf_len):
      buf.append(i)
    self.assertEqual(len(buf), buf_len)
    for i in range(buf_len):
      self.assertEqual(buf[i], i)

  def test_fill_buffer_more(self):
    buf_len = 100
    buf = RingBuffer(buf_len)
    for i in range(buf_len + 20):
      buf.append(i)

    self.assertEqual(len(buf), buf_len)
    for i in range(buf_len):
      self.assertEqual(buf[i], 20 + i)

  def test_get_last_index(self):
    buf_len = 100
    buf = RingBuffer(buf_len)
    for i in range(int(buf_len/2)):
      buf.append(i)

    self.assertEqual(buf.get_last_index(), buf_len/2 - 1)
    self.assertEqual(buf[buf.get_last_index()], 49)

    for i in range(buf_len):
      buf.append(i)

    self.assertEqual(buf.get_last_index(), buf_len - 1)
    self.assertEqual(buf[buf.get_last_index()], 99)

class ExperienceMemoryTestCase(unittest.TestCase):
  def test_save_experience_simple(self):
    mem = ExperienceMemory(memory_length=100)
    for i in range(4):
      mem.save_experience(i,i,i,False)

    for i in range(4):
      self.assertEqual(mem.observations[i], i)
      self.assertEqual(mem.actions[i], i)
      self.assertEqual(mem.rewards[i], i)
      self.assertEqual(mem.terminals[i], False)

  def test_save_experience_more(self):
    mem = ExperienceMemory(memory_length=100)

    for i in range(120):
      mem.save_experience(i,i,i,False)

    for i in range(100):
      self.assertEqual(mem.observations[i], i + 20)
      self.assertEqual(mem.actions[i], i + 20)
      self.assertEqual(mem.rewards[i], i + 20)
      self.assertEqual(mem.terminals[i], False)

  def test_get_exp_window(self):
    mem = ExperienceMemory(100)
    for i in range(10):
      mem.save_experience(i,i,i,False)
    mem.save_experience(66,6,6,True)
    self.assertEqual(mem.get_exp_window(10, 5), [6,7,8,9,66])

    # window should be padded with the first observation
    mem.save_experience(70,7,7,False)
    self.assertEqual(mem.get_exp_window(11, 5), [70,70,70,70,70])
    self.assertEqual(mem.get_exp_window(6, 5), [2,3,4,5,6])

  def test_get_current_window(self):
    mem = ExperienceMemory(memory_length=100)
    for i in range(4):
      mem.save_experience(i,i,i,False)
    ob = mem.get_current_window(4)
    self.assertEqual(ob, [0,1,2,3])

    mem.save_experience(1,2,3,True)
    ob = mem.get_current_window(4)
    self.assertEqual(ob, [1,2,3,1])

    mem.save_experience(1,2,3,False)
    ob = mem.get_current_window(4)
    self.assertEqual(ob, [1,1,1,1])

    for i in range(120):
      mem.save_experience(i,i,i,False)

    ob = mem.get_current_window(4)
    self.assertEqual(ob, [116,117,118,119])

  def test_sample_minibatch(self):
    mem = ExperienceMemory(memory_length=100)
    for i in range(6):
      mem.save_experience(i,i,i,False)

    mbo1, mba, mbr, mbo2, term = mem.sample_minibatch(3,2)
    self.assertEqual(len(mbo1), 6)
    self.assertEqual(len(mba), 3)

    # The last observations are sampled for every minibatch
    mem.save_experience(8,1,1,True)
    mem.save_experience(9,1,1,False)
    mem.save_experience(10,1,1,False)
    mbo1, mba, mbr, mbo2, term = mem.sample_minibatch(3,5)
    self.assertEqual(mbo1[-1], 9)
    self.assertEqual(mbo2[-1], 10)
