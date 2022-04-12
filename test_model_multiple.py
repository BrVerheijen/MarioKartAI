
import retro

from stable_baselines3.ppo import PPO

from mario_wrappers import Discretizer
from utils import SaveOnBestTrainingRewardCallBack, TimeLimitWrapper
from mario_wrappers import *
from retro_wrappers import wrap_deepmind_retro

import argparse

#Sets up terminal argument "load"
parser = argparse.ArgumentParser()
parser.add_argument('--load')
args = parser.parse_args()

#Variables should be the same as the ones used to train the model
workers = 4
steps = 2000
scenario = 'scenario'
state = retro.State.DEFAULT

#How many runs should be executed
RUNS = 5

#Creates a wrapper for the environment
#Discretizer: creates actions for the AI to use in the environment
def wrapper(env):
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    env = TimeLimitWrapperMarioKart(env, minutes=3, seconds=0)
    env = CutMarioMap(env, show_map=False)
    env = wrap_deepmind_retro(env)
    return env

#sets up retro environment
env = retro_make_vec_env('SuperMarioKart-Snes', scenario=scenario, state=state, n_envs=1,
                            vec_env_cls=lambda x: x[0](), max_episode_steps=4000,
                            wrapper_class=wrapper, seed=0, record=True)

#loads model with the argument passed to load
model = PPO.load(args.load)

obs = env.reset()

#sets up time variable which consists of 3 floats
time = (float("inf"), float("inf"), float("inf"))
for i in range(RUNS):
    sum_reward = 0

    #Render model and print run succesion, reward and time
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
