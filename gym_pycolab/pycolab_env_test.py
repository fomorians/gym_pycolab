"""Tests for pycolab environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from gym import spaces

from pycolab.examples.classics import four_rooms

from gym_pycolab import pycolab_env


class CustomColorsFourRoomsEnv(pycolab_env.PyColabEnv):
    """Classic four rooms game, with custom colors.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/classics/four_rooms.py
    """

    def __init__(self,
                 max_iterations=10,
                 default_reward=-1.):
        super(CustomColorsFourRoomsEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=8)

    def make_game(self):
        return four_rooms.make_game()

    def make_colors(self):
        return {
            'P': (0, 0, 255),
            ' ': (255, 0, 0),
            '#': (0, 255, 0),
        }


class PyColabEnvTest(parameterized.TestCase):

    def setUp(self):
        super(PyColabEnvTest, self).setUp()
        self._max_iterations = 10
        self._default_reward = 0

    def testBadMaxIterationsConstructor(self):
        with self.assertRaises(AssertionError):
            _ = CustomColorsFourRoomsEnv(
                max_iterations=-1,
                default_reward=self._default_reward)

    def testBadDefaultRewardConstructor(self):
        with self.assertRaises(AssertionError):
            _ = CustomColorsFourRoomsEnv(
                max_iterations=self._max_iterations,
                default_reward=None)

    @parameterized.parameters(
        ((13, 13, 3),),)
    def testReset(self, shape):
        env = CustomColorsFourRoomsEnv(
            max_iterations=self._max_iterations,
            default_reward=self._default_reward)
        last_state = env.reset()
        self.assertEqual(shape, last_state.shape)

    @parameterized.parameters(
        (1, (0, 1, 2)),
        (2, (0, 1)),
        (10, (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0)))
    def testMaxIterations(self, max_iterations, actions):
        env = CustomColorsFourRoomsEnv(
            max_iterations=max_iterations,
            default_reward=self._default_reward)

        _ = env.reset()
        for step, action in enumerate(actions):
            _, _, done, _ = env.step(action)
            if (step + 1) >= max_iterations:
                self.assertEqual(done, True)

    @parameterized.parameters(
        (0., 0., (0, 1, 2)),
        (-1., -3, (0, 1, 2)),
        (0., 1., (0, 0, 0, 0, 3, 3, 0, 0, 0, 2)))
    def testTotalRewards(self, default_reward, expected_total_reward, actions):
        env = CustomColorsFourRoomsEnv(
            max_iterations=self._max_iterations,
            default_reward=default_reward)

        _ = env.reset()
        total_reward = 0.
        for step, action in enumerate(actions):
            state, reward, _, _ = env.step(action)
            total_reward += reward
        env.close()
        self.assertEqual(total_reward, expected_total_reward)


if __name__ == '__main__':
    absltest.main()
