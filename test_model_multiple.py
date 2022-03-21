
import retro

from stable_baselines3.ppo import PPO

from mario_wrappers import Discretizer
from utils import SaveOnBestTrainingRewardCallBack, TimeLimitWrapper
from mario_wrappers import *
from retro_wrappers import wrap_deepmind_retro

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load')
args = parser.parse_args()

workers = 4
steps = 2000

RUNS = 5

state = retro.State.DEFAULT

def wrapper(env):
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    env = TimeLimitWrapperMarioKart(env, minutes=3, seconds=0)
    env = CutMarioMap(env, show_map=False)
    env = wrap_deepmind_retro(env)
    return env

env = retro_make_vec_env('SuperMarioKart-Snes', scenario='scenario', state=state, n_envs=1,
                            vec_env_cls=lambda x: x[0](), max_episode_steps=4000,
                            wrapper_class=wrapper, seed=0, record=True)

model = PPO.load(args.load)

obs = env.reset()

time = (float("inf"), float("inf"), float("inf"))
for i in range(RUNS):
    sum_reward = 0

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        sum_reward += reward

        env.render()

        if done:

            laps = env.data.lookup_value("lap") - 128
            min = env.data.lookup_value("currMin")
            sec = env.data.lookup_value("currSec")
            ms = ((env.data.lookup_value("currMiliSec") - 300) % 10000) / 100
            currTime = (min, sec, ms)

            if (laps >= 5 and currTime < time):
                time = currTime
                bestRun = i

            else: 
                print("FAIL")

            
            print("Run %d Final time:%2d:%2d:%2d Total Reward: %f" % (i, min, sec, ms, sum_reward))
            env.reset()
            break
