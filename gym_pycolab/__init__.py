from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

from gym_pycolab.pycolab_env import PyColabEnv
from gym_pycolab.pycolab_games import OrdealEnv
from gym_pycolab.pycolab_games import WarehouseManagerEnv
from gym_pycolab.pycolab_games import FluvialNatationEnv
from gym_pycolab.pycolab_games import ChainWalkEnv
from gym_pycolab.pycolab_games import CliffWalkEnv
from gym_pycolab.pycolab_games import FourRoomsEnv
from gym_pycolab.pycolab_games import ExtraterrestrialMaraudersEnv
from gym_pycolab.pycolab_games import ShockWaveEnv
from gym_pycolab.pycolab_games import ApertureEnv
from gym_pycolab.pycolab_games import ApprehendEnv
from gym_pycolab.pycolab_games import BetterScrollyMazeEnv


register(
    id='ChainWalk-v0',
    entry_point='gym_pycolab.pycolab_games:ChainWalkEnv',
    kwargs={'max_steps': 500})
register(
    id='CliffWalk-v0',
    entry_point='gym_pycolab.pycolab_games:CliffWalkEnv',
    kwargs={'max_steps': 500})
register(
    id='FourRooms-v0',
    entry_point='gym_pycolab.pycolab_games:FourRoomsEnv',
    kwargs={'max_steps': 500})
register(
    id='ExtraterrestrialMarauders-v0',
    entry_point='gym_pycolab.pycolab_games:ExtraterrestrialMaraudersEnv',
    kwargs={'max_steps': 500})
register(
    id='ShockWave-v0',
    entry_point='gym_pycolab.pycolab_games:ShockWaveEnv',
    kwargs={'level': 0, 'max_steps': 500})
register(
    id='ShockWave-v1',
    entry_point='gym_pycolab.pycolab_games:ShockWaveEnv',
    kwargs={'level': -1, 'max_steps': 500})
register(
    id='Aperture-v0',
    entry_point='gym_pycolab.pycolab_games:ApertureEnv',
    kwargs={'level': 0, 'max_steps': 500})
register(
    id='Aperture-v1',
    entry_point='gym_pycolab.pycolab_games:ApertureEnv',
    kwargs={'level': 1, 'max_steps': 500})
register(
    id='Aperture-v2',
    entry_point='gym_pycolab.pycolab_games:ApertureEnv',
    kwargs={'level': 2, 'max_steps': 500})
register(
    id='Apprehend-v0',
    entry_point='gym_pycolab.pycolab_games:ApprehendEnv',
    kwargs={'max_steps': 500})
register(
    id='FluvialNatation-v0',
    entry_point='gym_pycolab.pycolab_games:FluvialNatationEnv',
    kwargs={'max_steps': 500})
register(
    id='BetterScrollyMaze-v0',
    entry_point='gym_pycolab.pycolab_games:BetterScrollyMazeEnv',
    kwargs={'level': 0, 'max_steps': 500})
register(
    id='BetterScrollyMaze-v1',
    entry_point='gym_pycolab.pycolab_games:BetterScrollyMazeEnv',
    kwargs={'level': 1, 'max_steps': 500})
register(
    id='BetterScrollyMaze-v2',
    entry_point='gym_pycolab.pycolab_games:BetterScrollyMazeEnv',
    kwargs={'level': 2, 'max_steps': 500})
register(
    id='Ordeal-v0',
    entry_point='gym_pycolab.pycolab_games:OrdealEnv',
    kwargs={'max_steps': 500})
register(
    id='WarehouseManager-v0',
    entry_point='gym_pycolab.pycolab_games:WarehouseManagerEnv',
    kwargs={'level': 0, 'max_steps': 500})
register(
    id='WarehouseManager-v1',
    entry_point='gym_pycolab.pycolab_games:WarehouseManagerEnv',
    kwargs={'level': 1, 'max_steps': 500})
register(
    id='WarehouseManager-v2',
    entry_point='gym_pycolab.pycolab_games:WarehouseManagerEnv',
    kwargs={'level': 2, 'max_steps': 500})
