#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
created 23.04.2021
@author: holger mohr
"""

import torch
import numpy as np
from ale_python_interface import ALEInterface


def main():
    
    game_path = '/home/mohr/pytorch-dqn/container_mount/roms/space_invaders.bin'
    
    ale = ALEInterface()
    
    ale.setInt("random_seed", 0)
    
    getMinimalActionSet
    
    save_checkpoints_path = '/home/mohr/pytorch-dqn/container_mount/checkpoints'
    
    no_of_episodes = 10
    
    for episode_index in range(no_of_episodes)
    

if __name__ == '__main__':
    main()
