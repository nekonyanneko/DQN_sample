#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from gym.envs.registration import register


register(
    id='myenv-v0',
    entry_point='myenv.sampleEnv:MyEnv'
    )
register(
    id='myenv-v1',
    entry_point='myenv.env:MyEnv'
    )
register(
    id='myenv-v2',
    entry_point='myenv.envAd:MyEnv'
    )
