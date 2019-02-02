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
    FIELD_TYPES = [
        'S',  # 0: Start
        'G',  # 1: Goal
        '~',  # 2: 芝生(Damageを受ける確率1/10)
        'w',  # 3: 森(Damageを受ける確率1/2)
        '=',  # 4: 毒沼(1step毎に1のダメージ)
        'A',  # 5: 山(歩けない)
        'Y',  # 6: 勇者
        'M',  # 7: 敵
    ]
    MAP = np.array([
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [5, 5, 2, 0, 2, 2, 5, 2, 2, 4, 2, 2],
        [5, 2, 2, 2, 2, 2, 5, 5, 4, 4, 2, 2],
        [2, 2, 3, 3, 3, 3, 5, 5, 2, 2, 3, 3],
        [2, 3, 3, 3, 3, 5, 2, 2, 1, 2, 2, 3],
        [2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2],
    ])
    MAX_STEPS = 100
    MAX_DAMAGE = 100

    def __init__(self):
        """action空間と観測空間、報酬のmin,maxのリスト"""
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD_TYPES),
            shape=self.MAP.shape
            )
        self.reward_range = [-1., 100.]
        self._reset()

    def _reset(self):
        """状態を初期化し、初期の観測値を返す"""
        self.pos = self._find_pos('S')[0]
        self.goal = self._find_pos('G')[0]
        self.mon_pos = self._find_pos('G')[0]
        self.done = False
        self.damage = 0
        self.steps = 0

        return self._observe()

    def _next_move(self, pos, action, mon=False):
        # 1stepの処理
        if action == 0:
            next_pos = pos + [0, 1]
        elif action == 1:
            next_pos = pos + [0, -1]
        elif action == 2:
            next_pos = pos + [1, 0]
        elif action == 3:
            next_pos = pos + [-1, 0]

        if self._is_movable(next_pos, mon=mon):
            moved = True
        else:
            moved = False

        return next_pos, moved

    def _step(self, action):
        """actionを実行し、結果を返す"""
        pos, moved = self._next_move(self.pos, action)
        self.pos = pos if moved is True else self.pos

        mon_action = self.action_space.sample()
        mon_pos, mon_moved = self._next_move(self.mon_pos, mon_action,
                                             mon=True)
        self.mon_pos = mon_pos if mon_moved is True else self.mon_pos

        observation = self._observe()
        reward = self._get_reward(self.pos, moved)
        self.damage += self._get_damage(self.pos)
        self.done = self._is_done()

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

    def _get_reward(self, pos, moved):
        """報酬の計算"""
        if moved and (self.goal == pos).all():
            return max(100 - self.damage, 0)
        else:
            return -1

    def _get_damage(self, pos):
        """ダメージ計算"""
        field_type = self.FIELD_TYPES[self.MAP[tuple(pos)]]
        if field_type == 'S' or 'G':
            return 0
        elif field_type == '~':
            return 10 if np.random.random() < 1/10. else 0
        elif field_type == 'w':
            return 10 if np.random.random() < 1/2. else 0
        elif field_type == '=':
            return 1
        elif field_type == 'M':
            return self.MAX_DAMAGE + 1

    def _is_movable(self, pos, mon=False):
        """移動できるかの確認"""
        if mon is False:
            return (
                0 <= pos[0] < self.MAP.shape[0]
                and 0 <= pos[1] < self.MAP.shape[1]
                and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'A'
            )
        else:
            return (
                0 <= pos[0] < self.MAP.shape[0]
                and 0 <= pos[1] < self.MAP.shape[1]
                and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'A'
                and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'G'
            )

    def _observe(self):
        """マップに勇者の位置を重ねて返す"""
        observation = self.MAP.copy()
        observation[tuple(self.pos)] = self.FIELD_TYPES.index('Y')
        observation[tuple(self.mon_pos)] = self.FIELD_TYPES.index('M')

        return observation

    def _is_done(self):
        if (self.pos == self.goal).all():
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
