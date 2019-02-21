from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import gym
from gym import spaces
from gym import logger
from gym.utils import seeding
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
                 action_space=None,
                 delay=30,
                 resize_scale=8,
                 colors=None,
                 observation_type='layers'):
        """Create an `PyColabEnv` adapter to a `pycolab` game as a `gym.Env`.

        You can access the `pycolab.Engine` instance with `env.current_game`.

        Args:
            game_factory: function that creates a new `pycolab` game
            max_iterations: maximum number of steps.
            default_reward: default reward if reward is None returned by the
                `pycolab` game.
            action_space: the action `Space` of the environment.
            delay: renderer delay.
            resize_scale: number of pixels per observation pixel.
        """
        self._game_factory = game_factory
        self._max_iterations = max_iterations
        self._default_reward = default_reward
        self._colors = colors
        self.np_random = None

        test_game = self._game_factory()
        setattr(test_game.the_plot, 'info', {})
        observations, _, _ = test_game.its_showtime()
        layers = list(observations.layers.keys())
        not_ordered = list(set(layers) - set(test_game.z_order))

        self._render_order = list(reversed(not_ordered + test_game.z_order))
        self._observation_order = sorted(layers)
        self._observation_type = observation_type        

        if self._observation_type == 'layers':
            channels = [len(layers)]
            channel_max = 1.
        elif self._observation_type == 'labels':
            channels = [1]
            channel_max = 1.
        elif self._observation_type == 'rgb':
            assert self._colors is not None, ''
            channels = [3]
            channel_max = 255.
        else:
            raise NotImplementedError(self._observation_type)

        self._game_shape = list(observations.board.shape) + channels
        self.observation_space = spaces.Box(
            low=np.zeros(self._game_shape, np.float32), 
            high=np.ones(self._game_shape, np.float32) * channel_max, 
            dtype=np.float32)
        self.action_space = action_space

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
        # TODO(wenkesj) cache this control flow.
        if self._observation_type == 'layered':
            return np.stack([
                np.asarray(observations.layers[layer_key], np.float32) 
                for layer_key in self._observation_order], axis=-1)
        elif self._observation_type == 'labels':
            board = np.zeros(self._game_shape, np.int32)
            board_mask = np.zeros(self._game_shape, np.int32)
            for key in self._render_order:
                board_layer_mask = np.array(observations.layers[key]) * self._observation_order.index(key)
                board = np.where(np.logical_not(board_mask), board_layer_mask, board)
                board_mask = np.logical_or(board_layer_mask, board_mask)
            return board.astype(np.float32)
        elif self._observation_type == 'rgb':
            return self.paint_image(observations.layers)

    def _paint_board(self, layers):
        board_shape = self._last_observations.board.shape
        board = np.zeros(list(board_shape) + [3], np.uint32)
        board_mask = np.zeros(list(board_shape) + [3], np.bool)

        for key in self._render_order:
            color = self._colors.get(key, (0, 0, 0))
            color = np.reshape(color, [1, 1, -1]).astype(np.uint32)

            # Broadcast the layer to [H, W, C].
            board_layer_mask = np.array(layers[key])[..., None]
            board_layer_mask = np.repeat(board_layer_mask, 3, axis=-1)

            # Update the board with the new layer.
            board = np.where(np.logical_not(board_mask), board_layer_mask * color, board)

            # Update the mask.
            board_mask = np.logical_or(board_layer_mask, board_mask)
        return board

    def paint_image(self, layers, board=None):
        """Paint the layers into the board and return an RGB array."""
        if self._colors:
            img = self._paint_board(layers)
        else:
            assert board is not None, '`board` must not be `None` if there are no colors.'
            img = board
        img = np.repeat(np.repeat(
            img, self.resize_scale, axis=0), self.resize_scale, axis=1)
        if len(img.shape) != 3:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img.astype(np.uint8)
        return img

    def layers_from_board(self, board):
        """Convert the observation board into into layers."""
        board_layers = np.rollaxis(board, axis=-1)
        layers = {}
        for idx, key in enumerate(self._observation_order):
            layers[key] = board_layers[idx]
        return layers

    def _update_for_game_step(self, observations, reward):
        """Update internal state with data from an environment interaction."""
        self._last_observations = observations
        self._empty_board = np.zeros_like(self._last_observations.board)
        self._last_state = self.get_states(observations)
        self._last_reward = reward if reward is not None else self._default_reward
        self._game_over = self.current_game.game_over

        if self.current_game.the_plot.frame >= self._max_iterations:
            self._game_over = True

    def reset(self):
        """Start a new episode."""
        self.current_game = self._game_factory()
        setattr(self.current_game.the_plot, 'info', {})
        self._game_over = None
        self._last_observations = None
        self._last_reward = None
        observations, reward, _ = self.current_game.its_showtime()
        self._update_for_game_step(observations, reward)
        return self._last_state

    def step(self, action):
        """Apply action, step the world forward, and return observations."""
        if self.current_game is None:
            logger.warn("Episode has already ended, call `reset` instead..")
            state = self._last_state
            reward = self._last_reward
            done = self._game_over
            return state, reward, done, {}

        # Execute the action in pycolab.
        setattr(self.current_game.the_plot, 'info', {})
        observations, reward, _ = self.current_game.play(action)
        self._update_for_game_step(observations, reward)
        info = getattr(self.current_game.the_plot, 'info', {})

        # Check the current status of the game.
        state = self._last_state
        reward = self._last_reward
        done = self._game_over

        if self._game_over:
            self.current_game = None
        return state, reward, done, info

    def render(self, mode='human'):
        """Render the board to the gym viewer."""
        if self._last_observations:
            img = self._last_observations.board
            layers = self._last_observations.layers
            img = self.paint_image(layers, board=img)
        else:
            img = self._empty_board
            img = np.repeat(np.repeat(
                img, self.resize_scale, axis=0), self.resize_scale, axis=1)
            if len(img.shape) != 3:
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
