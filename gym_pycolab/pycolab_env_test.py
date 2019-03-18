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


class PyColabEnvTest(parameterized.TestCase):

    def setUp(self):
        super(PyColabEnvTest, self).setUp()
        self._game_factory = four_rooms.make_game
        self._action_space = spaces.Discrete(4 + 1)
        self._max_iterations = 10
        self._default_reward = 0
        self._resize_scale = 8
        self._colors = {
            'P': (0, 0, 255), 
            ' ': (255, 0, 0), 
            '#': (0, 255, 0),
        }
        self._observation_type = 'layers'
  
    def testBadObservationTypeConstructor(self):
        with self.assertRaises(AssertionError):
            _ = pycolab_env.PyColabEnv(
                game_factory=self._game_factory,
                action_space=self._action_space,
                max_iterations=self._max_iterations,
                default_reward=self._default_reward,
                resize_scale=self._resize_scale,
                observation_type='bad_observation_type')

    def testBadMaxIterationsConstructor(self):
        with self.assertRaises(AssertionError):
            _ = pycolab_env.PyColabEnv(
                game_factory=self._game_factory,
                action_space=self._action_space,
                max_iterations=-1,
                default_reward=self._default_reward,
                resize_scale=self._resize_scale,
                observation_type=self._observation_type)

    def testBadDefaultRewardConstructor(self):
        with self.assertRaises(AssertionError):
            _ = pycolab_env.PyColabEnv(
                game_factory=self._game_factory,
                action_space=self._action_space,
                max_iterations=self._max_iterations,
                default_reward=None,
                resize_scale=self._resize_scale,
                observation_type=self._observation_type)

    @parameterized.named_parameters(
        ('Layers', 'layers', (13, 13, 3)),
        ('Labels', 'labels', (13, 13)),
        ('RGB', 'rgb', (13, 13, 3)))
    def testObservationType(self, observation_type, shape):
        env = pycolab_env.PyColabEnv(
            game_factory=self._game_factory,
            action_space=self._action_space,
            max_iterations=self._max_iterations,
            default_reward=self._default_reward,
            colors=self._colors,
            resize_scale=self._resize_scale,
            observation_type=observation_type)
        last_state = env.reset()
        self.assertEqual(shape, last_state.shape)

    @parameterized.parameters(
        (1, (0, 1, 2)), 
        (2, (0, 1)), 
        (10, (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0)))
    def testMaxIterations(self, max_iterations, actions):
        env = pycolab_env.PyColabEnv(
            game_factory=self._game_factory,
            action_space=self._action_space,
            max_iterations=max_iterations,
            default_reward=self._default_reward,
            resize_scale=self._resize_scale,
            observation_type=self._observation_type)

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
        env = pycolab_env.PyColabEnv(
            game_factory=self._game_factory,
            action_space=self._action_space,
            max_iterations=self._max_iterations,
            default_reward=default_reward,
            resize_scale=self._resize_scale,
            observation_type=self._observation_type)

        _ = env.reset()
        total_reward = 0.
        for step, action in enumerate(actions):
            state, reward, _, _ = env.step(action)
            total_reward += reward
        env.close()
        self.assertEqual(total_reward, expected_total_reward)


if __name__ == '__main__':
    absltest.main()