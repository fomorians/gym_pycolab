from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import gym
from gym import spaces
import numpy as np


class PyColabEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, 
                 game_factory, 
                 max_iterations, 
                 default_reward,
                 num_actions=4,
                 delay=30,
                 resize_scale=8):
        """Create an `PyColabEnv` adapter to a `pycolab` game as a `gym.Env`.

        You can access the `pycolab.Engine` instance with `env.current_game`.

        Args:
            game_factory: function that creates a new `pycolab` game
            max_iterations: maximum number of steps.
            default_reward: default reward if reward is None returned by the
                `pycolab` game.
            num_actions: number of possible actions
            delay: renderer delay.
            resize_scale: number of pixels per observation pixel.
        """
        self._game_factory = game_factory
        self._max_iterations = max_iterations
        self._default_reward = default_reward

        test_game = self._game_factory()
        observations, _, _ = test_game.its_showtime()
        layers = sorted(list(observations.layers.keys()))
        self._value_mapping = {k: v for v, k in enumerate(layers)}

        game_shape = observations.board.shape
        low = np.zeros(game_shape, np.uint32)
        high = np.full(
            game_shape, 
            max(list(self._value_mapping.values())), 
            np.uint32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint32)

        # TODO(wenkesj): handle action space better. 
        # There are some constructs that are multi-agent (list/dict, etc.)
        self.action_space = spaces.Discrete(num_actions)

        self.current_game = None   

        self._last_observations = None
        self._empty_board = None
        self._last_state = None
        self._last_reward = None
        self._game_over = False 

        self.viewer = None
        self.resize_scale = resize_scale
        self.delay = delay

    def get_states(self, observations):
        """Transform the pycolab `rendering.Observations` to a state."""
        # TODO(wenkesj): there might be a better way, 
        # this current function decreases fps by ~2x (compared to below).
        # 
        # This is the optimal scenario, so maybe there is a way to map values 
        #   ahead of time?
        # >>> return observations.board

        board = np.zeros_like(observations.board).astype(np.uint32)
        for value, layer in observations.layers.items():
            board_mask = np.array(layer, np.uint32) * self._value_mapping[value]
            board += board_mask
        return board

    def _update_for_game_step(self, observations, reward):
        """Update internal state with data from an environment interaction."""
        self._last_observations = observations
        self._empty_board = np.zeros_like(self._last_observations.board)
        self._last_state = self.get_states(observations)
        self._last_reward = reward if reward is not None else self._default_reward
        self._game_over = self.current_game.game_over

        if self.current_game.the_plot.frame >= self._max_iterations:
            self._game_over = True

    def _reset_game(self):
        """Clear all the internal information about the game."""
        self.current_game = None
        self._game_over = None
        self._last_observations = None
        self._last_reward = None

    def reset(self):
        """Start a new episode."""
        self.current_game = self._game_factory()
        observations, reward, _ = self.current_game.its_showtime()
        self._update_for_game_step(observations, reward)
        return self._last_state

    def step(self, action):
        """Apply action, step the world forward, and return observations."""
        if self.current_game is None:
            raise RuntimeError(
                "Episode has already ended, call `reset` instead..")

        # Execute the action in pycolab.
        observations, reward, _ = self.current_game.play(action)
        self._update_for_game_step(observations, reward)

        # Check the current status of the game.
        state = self._last_state
        reward = self._last_reward
        done = self._game_over
        info = {}

        if self._game_over:
            self._reset_game()
        return state, reward, done, info

    def render(self, mode='human'):
        """Render the board to the gym viewer."""
        # TODO(wenkesj): handle pycolab colors.
        if self._last_observations:
            img = self._last_observations.board
            img = np.repeat(np.repeat(
                img, self.resize_scale, axis=0), self.resize_scale, axis=1)
            img = np.repeat(img[..., None], 3, axis=-1)
        else:
            img = self._empty_board
            img = np.repeat(np.repeat(
                img, self.resize_scale, axis=0), self.resize_scale, axis=1)
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img.astype(np.uint8)

        if mode == 'rgb_array':
            return img

        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(self.delay / 1e3)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
