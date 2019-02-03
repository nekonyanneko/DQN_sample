#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import gym
import numpy as np
import gym.spaces
from time import sleep


class Player():
    def __init__(self, is_later):
        self.is_win = False
        self.action_count = 0
        self.is_later = is_later

    def is_threshold_exceeded(self, limit_num):
        if self.action_count >= limit_num:
            return True
        return False


class MyEnv(gym.Env):
    # human: 画面表示のため.戻り値なし
    # ansi: 文字列 or StringIOを返す
    metadata = {'render.modes': ['human', 'ansi']}
    FIELD_TYPES = [
        '○',
        '×',
        '-',
    ]
    MAP = np.array([
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ])
    INIT_MAP = np.array([
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ])
    WIN_MAP = np.array([
        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
        [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
        [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
        [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]],
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
        [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2]],
        [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3]],
        [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]]
    ])
    PREEMPTION_MAX_STEPS = 13
    LATE_MAX_STEPS = 12

    def __init__(self):
        """action空間と観測空間、報酬のmin,maxのリスト"""
        super().__init__()
        self.action_space = gym.spaces.Discrete(25)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD_TYPES),
            shape=self.MAP.shape
            )
        self.MAP = self.INIT_MAP.copy()
        self.reward_range = [-2., 100.]
        self._reset()

    def _reset(self):
        print('reset')
        self.MAP = self.INIT_MAP.copy()
        print(self.MAP)
        print(self.INIT_MAP)
        """状態を初期化し、初期の観測値を返す"""
        self.preemption_player = Player(False)
        self.late_player = Player(True)
        self.done = False
        self.steps = 0
        return self._observe()

    def _step(self, action):
        # 1stepの処理
        def action_coordinate(act):
            x_coord = (act // 5) - 1
            y_coord = act % 5
            return [x_coord, y_coord]

        action_coord = action_coordinate(action)

        if self.steps % 2 == 0:
            is_late = False
        else:
            is_late = True

        is_miss = self._set_coord(action_coord, is_late)

        if is_late:
            self.late_player.action_count += 1
        else:
            self.preemption_player.action_count += 1

        observation = self._observe()
        reward = self._get_reward(is_late, observation, is_miss)
        self.steps += 1
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

    def _get_reward(self, is_late, observation, is_miss):
        """報酬の計算"""
        # マップから勝敗の確認
        if is_late:
            pos = self._find_pos('×')
            print(pos)
            print(np.all(self.WIN_MAP == pos))
            if np.all(pos == self.WIN_MAP):
                sleep(2)
                self.late_player.is_win = True
        else:
            pos = self._find_pos('○')
            if np.all(pos == self.WIN_MAP):
                sleep(2)
                self.preemption_player.is_win = True

        # 報酬計算
        if is_miss:
            return -2
        elif self.preemption_player.is_win:
            return max(100 - self.preemption_player.action_count, 0)
        elif self.late_player.is_win:
            return -1
        elif not self.done:
            return 0
        else:
            return -1

    def _is_done(self):
        if self.preemption_player.is_win or \
           self.late_player.is_win or \
           self._is_draw():
            return True
        return False

    def _is_draw(self):
        if self.preemption_player.is_threshold_exceeded(
            self.PREEMPTION_MAX_STEPS) and \
           self.late_player.is_threshold_exceeded(self.LATE_MAX_STEPS):
            return True
        return False

    def _observe(self):
        """マップに勇者の位置を重ねて返す"""
        observation = self.MAP.copy()
        return observation

    def _set_coord(self, coord, is_late):
        is_miss = False
        if is_late:
            if self.FIELD_TYPES[self.MAP[tuple(coord)]] != '-':
                self.preemption_player.is_win = True
                print('miss')
                is_miss = True
            self.MAP[tuple(coord)] = self.FIELD_TYPES.index('×')
        else:
            if self.FIELD_TYPES[self.MAP[tuple(coord)]] != '-':
                self.late_player.is_win = True
                print('miss')
                is_miss = True
            self.MAP[tuple(coord)] = self.FIELD_TYPES.index('○')
        return is_miss

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
            self.MAP == self.FIELD_TYPES.index(field_type)
            ))))
