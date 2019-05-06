from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages


setup(
    name='gym_pycolab',
    version='0.0.0',
    description='Gym interface for custom pycolab games.',
    url='https://github.com/fomorians/gym_pycolab',
    packages=find_packages(),
    install_requires=['pycolab', 'gym'])
