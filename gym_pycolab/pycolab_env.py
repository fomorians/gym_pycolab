"""The pycolab environment interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numbers
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
                 observation_type='layers',
                 exclude_from_state=None,
                 merge_layer_groups=None):
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
                Used only by the renderer.
            colors: optional dictionary mapping key name to `tuple(R, G, B)` 
                or a callable that returns a dictionary mapping key name to 
                `tuple(R, G, B)`.
            observation_type: type of observations to return.
                - layers: the 3D board where each cell corresponds to a class 
                    represented by the objects in the game. If the object exists 
                    in the cell, the value will be 1, otherwise 0. If the game 
                    occludes layers, the rendering order is ignored and multiple
                    objects can be represented by 1. For example, Bridge = 1, 
                    Water = 2 -> [[[0, 1, 1], ...]], the Bridge and Water are 
                    represented.
                - labels: the 2D board where each cell corresponds to a class 
                    represented by the objects in the game. Only one object 
                    class is represented, following the rendering order of the 
                    game. For example, Bridge = 1, Water = 2 -> [[1, ...]],
                    the Bridge is represented instead of the Water.
                - rgb: TODO(wenkesj): docstring.
            exclude_from_state: set to exclude from the observations to states.
            merge_layer_groups: merge layers for these group of observations keys.
        """
        assert observation_type in ['layers', 'labels', 'rgb']
        assert max_iterations > 0
        assert isinstance(default_reward, numbers.Number)

        self._game_factory = game_factory
        self._max_iterations = max_iterations
        self._default_reward = default_reward
        if callable(colors):
            self._colors = None
            self._colors_factory = colors
        else:
            self._colors = colors
            self._colors_factory = None
        self.np_random = None

        test_game = self._game_factory()
        setattr(test_game.the_plot, 'info', {})
        observations, _, _ = test_game.its_showtime()
        layers = list(observations.layers.keys())
        not_ordered = list(set(layers) - set(test_game.z_order))

        self._render_order = list(reversed(not_ordered + test_game.z_order))

        if exclude_from_state is None:
            exclude_from_state = []
        self._exclude_from_state = set(exclude_from_state)

        if merge_layer_groups is None:
            merge_layer_groups = [set([])]
        self._merge_layer_groups = merge_layer_groups

        observation_layers = list(set(layers) - self._exclude_from_state)
        self._observation_order = sorted(observation_layers)
        self._observation_type = observation_type

        if self._observation_type == 'layers':
            merge_size_reduction = 0
            for group in self._merge_layer_groups:
                if group:
                    merge_size_reduction += len(group) - 1
            channels = [len(observation_layers) - merge_size_reduction]
            channel_max = 1.
            channel_min = 0.
            self.get_states = self._get_states_layers
        elif self._observation_type == 'labels':
            # TODO(wenkesj): implement merge_layer_groups
            channels = []
            channel_max = 1.
            channel_min = 0.
            self.get_states = self._get_states_labels
        elif self._observation_type == 'rgb':
            channels = [3]
            channel_max = 255.
            channel_min = 0.
            self.get_states = self._get_states_rgb

        self._game_shape = list(observations.board.shape) + channels
        self.observation_space = spaces.Box(
            low=np.ones(self._game_shape, np.float32) * channel_min, 
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

    def _get_states_layers(self, observations):
        """Transform the pycolab `rendering.Observations` to a state by layers."""
        layered_observations = np.stack([
            np.asarray(observations.layers[layer_key], np.float32) 
            for layer_key in self._observation_order], axis=-1)

        # TODO(wenkesj): is there a faster/better way to do this?
        #   We can probably precompute this and change the observation_order.
        if self._merge_layer_groups[0]:
            for group_set in self._merge_layer_groups:
                group = list(group_set)
                group_layers = []
                group_remove_indices = []

                leader_key = group[0]
                leader_layer_idx = self._observation_order.index(leader_key)
                leader_layer = layered_observations[..., leader_layer_idx]
                group_layers.append(leader_layer)
                
                for key in group[1:]:
                    layer_idx = self._observation_order.index(key)
                    group_remove_indices.append(layer_idx)
                    layer = layered_observations[..., layer_idx]
                    group_layers.append(layer)

                # remove layers that are merged.
                layered_observations[..., leader_layer_idx] = np.logical_or.reduce(group_layers)
                layered_observations = np.delete(layered_observations, group_remove_indices, -1)
        return layered_observations

    # TODO(wenkesj): implement merge_layer_groups for labels.
    def _get_states_labels(self, observations):
        """Transform the pycolab `rendering.Observations` to a state by label."""
        board = np.zeros(self._game_shape, np.int32)
        board_mask = np.zeros(self._game_shape, np.int32)

        for key in self._render_order:
            if (key in self._exclude_from_state):
                continue
            board_layer_mask = np.array(observations.layers[key]) * self._observation_order.index(key)
            board = np.where(np.logical_not(board_mask), board_layer_mask, board)
            board_mask = np.logical_or(board_layer_mask, board_mask)
        return board.astype(np.float32)

    def _get_states_rgb(self, observations):
        """Transform the pycolab `rendering.Observations` to a state by rgb."""
        board = self._paint_board(
            observations.layers, exclude=True).astype(np.float32)
        return board

    def _paint_board(self, layers, exclude=False):
        """Method to privately paint layers to RGB."""
        board_shape = self._last_observations.board.shape
        board = np.zeros(list(board_shape) + [3], np.uint32)
        board_mask = np.zeros(list(board_shape) + [3], np.bool)

        for key in self._render_order:
            if exclude and (key in self._exclude_from_state):
                continue

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

    def paint_image(self, layers, board=None, resize=True, exclude=False):
        """Paint the layers into the board and return an RGB array."""
        if self._colors:
            img = self._paint_board(layers, exclude=exclude)
        else:
            assert board is not None, '`board` must not be `None` if there are no colors.'
            img = board
        if resize:
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
        if self._colors_factory:
            self._colors = self._colors_factory()
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
            if self.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(self.delay / 1e3)
            return self.viewer.isopen

    def seed(self, seed=None):
        """Seeds the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Sets up the renderer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
