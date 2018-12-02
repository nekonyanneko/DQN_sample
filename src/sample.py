#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import myenv
import logger
import numpy as np
import argparse
from enum import Enum
import gym

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'myenv-v0'


class MYSTR(Enum):
    MYPROGRAM_NAME = 'MYPROGRAM'
    USAGE = 'DQN SAMPLE'
    DESCRIPTION = 'description'
    EPILOG = 'end'

    SEED = 123
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class OUTMSG(Enum):
    OUTPUT_HEADER = '[OUTPUT]: '
    ACTIONS_OP_MSG = 'actions are: '
    OBSERVE_OP_MSG = 'observetions are: '
    REWARDS_OP_MSG = 'rewards are: '


class ERRMSG(Enum):
    ERROR_HEADER = '[ERROR]: '
    EXEC_ERROR = 'Not supported such exec type.'


def myperser():
    parser = argparse.ArgumentParser(
                prog=MYSTR.MYPROGRAM_NAME.value,
                description=MYSTR.DESCRIPTION.value,
                epilog=MYSTR.EPILOG.value,
                add_help=True,
                )
    parser.add_argument('-s', '--nb-steps', default=50000,
                        help='step count')
    parser.add_argument('-l', '--limit', default=50000,
                        help='memory limit')
    parser.add_argument('-wl', '--window-length', default=1,
                        help='window length')
    parser.add_argument('-u', '--target-model-update', default=1e-2,
                        help='target models update rate')
    parser.add_argument('-r', '--learning-rate', default=1e-3,
                        help='deep learnings learning rate')
    parser.add_argument('-wu', '--warmup', default=10,
                        help='warmup number')
    parser.add_argument('-te', '--nb-episodes', default=5,
                        help='dqn test episodes')
    parser.add_argument('-e', '--exec-type', help='TRAIN or TEST')
    parser.add_argument('-v', '--verbose', default=2,
                        help='select mode')
    return parser


def create_model(env, action_n):
    in_ = Input(shape=(1,) + env.observation_space.shape, name='input')
    fl_ = Flatten()(in_)
    dn_ = Dense(16, activation='relu')(fl_)
    dn_ = Dense(16, activation='relu')(dn_)
    dn_ = Dense(16, activation='relu')(dn_)
    ou_ = Dense(action_n, activation='linear')(dn_)
    return Model(inputs=in_, outputs=ou_)


def exec_dqn(trainOrTest, env, dqn, nb_steps, verbose, episodes):
    if trainOrTest == MYSTR.TRAIN.value:
        dqn.fit(env, nb_steps=nb_steps,
                visualize=True,
                verbose=verbose)
        dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME),
                         overwrite=True)
    elif trainOrTest == MYSTR.TEST.value:
        dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
        cb_ep = logger.EpisodeLogger()
        dqn.test(env, nb_episodes=episodes, visualize=True,
                 callbacks=[cb_ep])
        for index, ep_ac in enumerate(cb_ep.actions.values()):
            print(OUTMSG.OUTPUT_HEADER.value +
                  'episode_{}:'.format(index) +
                  OUTMSG.ACTIONS_OP_MSG.value +
                  str(ep_ac))
    else:
        raise TypeError(ERRMSG.ERROR_HEADER.value +
                        ERRMSG.EXEC_ERROR.value)


def main():
    parser = myperser()
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    np.random.seed(MYSTR.SEED.value)
    env.seed(MYSTR.SEED.value)
    nb_actions = env.action_space.n

    model = create_model(env, nb_actions)
    print(model.summary())

    memory = SequentialMemory(limit=args.limit,
                              window_length=args.window_length)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=args.warmup,
                   target_model_update=args.target_model_update,
                   policy=policy)
    dqn.compile(Adam(lr=args.learning_rate), metrics=['mae'])

    exec_dqn(args.exec_type, env, dqn,
             args.nb_steps, args.verbose, args.nb_episodes)


if __name__ == '__main__':
    main()
