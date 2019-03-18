"""An example implementation of pycolab games as environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import aperture
from pycolab.examples import apprehend
from pycolab.examples import better_scrolly_maze
from pycolab.examples import extraterrestrial_marauders
from pycolab.examples import fluvial_natation
from pycolab.examples import ordeal
from pycolab.examples import shockwave
from pycolab.examples import warehouse_manager
from pycolab.examples.classics import chain_walk
from pycolab.examples.classics import cliff_walk
from pycolab.examples.classics import four_rooms

from gym_pycolab import pycolab_env


class OrdealEnv(pycolab_env.PyColabEnv):
    """Ordeal game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/ordeal.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(OrdealEnv, self).__init__(
            game_factory=ordeal.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(3 + 1),
            resize_scale=8)


class WarehouseManagerEnv(pycolab_env.PyColabEnv):
    """Warehouse manager game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/warehouse_manager.py
    """

    def __init__(self, 
                 level=0,
                 max_steps=10,
                 default_reward=-1.):
        super(WarehouseManagerEnv, self).__init__(
            game_factory=lambda: warehouse_manager.make_game(level), 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=8)


class FluvialNatationEnv(pycolab_env.PyColabEnv):
    """Fluvial natation game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/fluvial_natation.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(FluvialNatationEnv, self).__init__(
            game_factory=fluvial_natation.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(2 + 1),
            resize_scale=8)


class ChainWalkEnv(pycolab_env.PyColabEnv):
    """Classic chain walk game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/classics/chain_walk.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(ChainWalkEnv, self).__init__(
            game_factory=chain_walk.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(2 + 1),
            resize_scale=8)


class CliffWalkEnv(pycolab_env.PyColabEnv):
    """Classic cliff walk game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/classics/cliff_walk.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(CliffWalkEnv, self).__init__(
            game_factory=cliff_walk.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=8)


class FourRoomsEnv(pycolab_env.PyColabEnv):
    """Classic four rooms game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/classics/four_rooms.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(FourRoomsEnv, self).__init__(
            game_factory=four_rooms.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=8)


class ExtraterrestrialMaraudersEnv(pycolab_env.PyColabEnv):
    """Extraterrestrial marauders game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/extraterrestrial_marauders.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(ExtraterrestrialMaraudersEnv, self).__init__(
            game_factory=extraterrestrial_marauders.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(3 + 1),
            resize_scale=8)


class ShockWaveEnv(pycolab_env.PyColabEnv):
    """Shock wave game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/shockwave.py
    """

    def __init__(self, 
                 level=0,
                 max_steps=10,
                 default_reward=-1.):
        super(ShockWaveEnv, self).__init__(
            game_factory=lambda: shockwave.make_game(level), 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(3 + 1),
            resize_scale=8)


class ApertureEnv(pycolab_env.PyColabEnv):
    """Aperature game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/aperture.py
    """

    def __init__(self, 
                 level=0,
                 max_steps=10,
                 default_reward=-1.):
        super(ApertureEnv, self).__init__(
            game_factory=lambda: aperture.make_game(level), 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(8 + 1),
            resize_scale=16)


class ApprehendEnv(pycolab_env.PyColabEnv):
    """Apprehend game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/apprehend.py
    """

    def __init__(self, 
                 max_steps=10,
                 default_reward=-1.):
        super(ApprehendEnv, self).__init__(
            game_factory=apprehend.make_game, 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(2 + 1),
            resize_scale=8)


class BetterScrollyMazeEnv(pycolab_env.PyColabEnv):
    """Better scrolly maze game.

    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/better_scrolly_maze.py
    """

    def __init__(self, 
                 level=0,
                 max_steps=10,
                 default_reward=-1.):
        super(BetterScrollyMazeEnv, self).__init__(
            game_factory=lambda: better_scrolly_maze.make_game(level), 
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=8)


if __name__ == "__main__":
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--game', 
        choices=[
            'chain_walk', 
            'cliff_walk', 
            'four_rooms',
            'extraterrestrial_marauders',
            'shockwave',
            'aperture',
            'apprehend',
            'better_scrolly_maze',
            'ordeal',
            'fluvial_natation',
            'warehouse_manager'], 
        required=True)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    if args.game == 'chain_walk':
        env = ChainWalkEnv(max_steps=250)
    elif args.game == 'cliff_walk':
        env = CliffWalkEnv(max_steps=250)
    elif args.game == 'four_rooms':
        env = FourRoomsEnv(max_steps=250)
    elif args.game == 'extraterrestrial_marauders':
        env = ExtraterrestrialMaraudersEnv(max_steps=250)
    elif args.game == 'shockwave':
        env = ShockWaveEnv(max_steps=250)
    elif args.game == 'aperture':
        env = ApertureEnv(max_steps=250)
    elif args.game == 'apprehend':
        env = ApprehendEnv(max_steps=250)
    elif args.game == 'better_scrolly_maze':
        env = BetterScrollyMazeEnv(max_steps=250)
    elif args.game == 'ordeal':
        env = OrdealEnv(max_steps=250)
    elif args.game == 'warehouse_manager':
        env = WarehouseManagerEnv(max_steps=250)
    elif args.game == 'fluvial_natation':
        env = FluvialNatationEnv(max_steps=250)

    if args.benchmark:
        import time
        num_eps = 500
        total_eps_time = 0.
        total_fps = 0.
        for _ in range(num_eps):
            start = time.time()
            state = env.reset()
            done = False
            num_frames = 0
            while not done:
                _, _, done, _ = env.step(env.action_space.sample())
                num_frames += 1
            eps_time = (time.time() - start)
            total_eps_time += eps_time
            total_fps += (num_frames / eps_time)
        average_eps_time = total_eps_time / num_eps
        average_fps = total_fps / num_eps
        print('total eps: {}ms, avg. eps: {}ms, avg. fps: {}fps'.format(
            total_eps_time * 1e3, 
            average_eps_time * 1e3, 
            average_fps))
    else:
        state = env.reset()
        done = False
        env.render()
        while not done:
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()
        env.close()
