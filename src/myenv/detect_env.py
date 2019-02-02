#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import gym
import numpy as np
import gym.spaces


class MyEnv(gym.Env):
    # human: 画面表示のため.戻り値なし
    # ansi: 文字列 or StringIOを返す
    metadata = {'render.modes': ['human', 'ansi']}
    CORRECT = False
    MAX_STEPS = 100
    MAX_DAMAGE = 100
    MIN_DOMAIN = 1
    MAX_DOMAIN = 30
    MAP_LENGTH = 60
    CORRECT_EPSILON = 0.3
    CLEAN_EPSILON = 0.3
    TARGET_EPSILON = 1.0

    FIELD_TYPES = [
        '0',
        '1',
    ]
    MAP = np.zeros((MAP_LENGTH, MAX_DOMAIN), dtype=int)
    TARGET = None

    def __init__(self):
        """action空間と観測空間、報酬のmin,maxのリスト"""
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self._create_map_and_field()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD_TYPES),
            shape=self.MAP.shape
            )
        self.reward_range = [-100., 100.]
        self._reset()

    def _create_map_and_field(self):
        unique_domain_num = np.random.randint(self.MIN_DOMAIN, self.MAX_DOMAIN)
        domain = np.arange(unique_domain_num)
        self.CORRECT = True\
            if np.random.rand() < self.CORRECT_EPSILON else False
        self.TARGET = np.random.choice(domain)\
            if self.CORRECT is True else None

        for index, dom in enumerate(domain):
            if dom == self.TARGET:
                self.MAP[:, index] = np.random.choice(
                            [0, 1], self.MAP_LENGTH,
                            p=[1 - self.TARGET_EPSILON, self.TARGET_EPSILON])
            else:
                self.MAP[:, index] = np.random.choice(
                            [0, 1], self.MAP_LENGTH,
                            p=[1 - self.CLEAN_EPSILON, self.CLEAN_EPSILON])

    def _reset(self):
        """状態を初期化し、初期の観測値を返す"""
        self.pos = np.array([[0]])
        self.goal = np.array([[1]])
        self.done = False
        self.damage = 0
        self.steps = 0

        return self._observe()

    def _next_move(self, pos, action):
        # 1stepの処理
        if action == 0:
            next_pos = pos + [0]
            moved = True
        elif action == 1:
            next_pos = pos + [1]
            moved = False

        return next_pos, moved

    def _step(self, action):
        """actionを実行し、結果を返す"""
        pos, moved = self._next_move(self.pos, action)
        self.pos = pos if moved is True else self.pos

        observation = self._observe()
        reward = self._get_reward(self.pos, action)
        self.damage += self._get_damage(action)
        self.done = self._is_done()
        print('damage: {}'.format(self.damage))
        print('pos: {}'.format(self.pos))
        print('action: {}'.format(action))
        print('target: {}'.format(self.TARGET))
        self._create_map_and_field()

        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        """環境を可視化する"""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n'.join(' '.join(
                    self.FIELD_TYPES[elem] for elem in row
                    ) for row in self._observe()
                ) + '\n\n'
            )

        return outfile

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self, pos, action):
        """報酬の計算"""
        if action == 0:
            if self.TARGET is not None:
                return -100
            else:
                return 25
        else:
            if self.TARGET is not None:
                return 100
            else:
                return -100

    def _get_damage(self, action):
        if action == 0:
            if self.TARGET is not None:
                return 25
            else:
                return 1
        else:
            if self.TARGET is not None:
                return 1
            else:
                return self.MAX_DAMAGE + 1

    def _observe(self):
        observation = self.MAP.copy()

        return observation

    def _is_done(self):
        if self.pos == [[1]]:
            return True
        elif self.steps > self.MAX_STEPS:
            return True
        elif self.damage > self.MAX_DAMAGE:
            return True
        else:
            return False

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
            self.MAP == self.FIELD_TYPES.index(field_type)
            ))))
