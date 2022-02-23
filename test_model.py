from ast import arg
import retro

from stable_baselines3.ppo import PPO
from stable_baselines3.common.cmd_util import make_vec_env

from mario_wrappers import Discretizer
from utils import SaveOnBestTrainingRewardCallBack, TimeLimitWrapper
from mario_wrappers import Discretizer, retro_make_vec_env, CutMarioMap, DiscretizerActions
from retro_wrappers import wrap_deepmind_retro

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load')
args = parser.parse_args()

workers = 4
steps = 1000

state=retro.State.DEFAULT
# state = ["BowserCastle_M", "BowserCastle2_M", "BowserCastle3_M", "ChocoIsland_M", "ChocoIsland2_M", "DonutPlains_M",
#           "DonutPlains2_M", "DonutPlains3_M", "GhostValley_M", "GhostValley2_M", "GhostValley3_M", "KoopaBeach_M",
#           "KoopaBeach2_M", "MarioCircuit_M", "MarioCircuit2_M", "MarioCircuit3_M", "MarioCircuit4_M", "RainbowRoad_M",
#           "VanillaLake_M", "VanillaLake2_M"]

def wrapper(env):
    env= Discretizer(env, DiscretizerActions.SIMPLE)

    env= CutMarioMap(env, show_map=False)
    env=wrap_deepmind_retro(env)
    return env

env = retro_make_vec_env('SuperMarioKart-Snes', scenario='training_check', state=state, n_envs=1,
                            vec_env_cls=lambda x: x[0](), max_episode_steps=4000,
                            wrapper_class=wrapper, seed=0, record=True)

model = PPO.load(args.load)

obs=env.reset()

sum_reward=0
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    sum_reward += reward

    print(action, reward)
    env.render()
    if done:
        print("total reward:", sum_reward)
        print("Final time: %2d:%2d:%2d\n" % (env.data.lookup_value("currMin"), env.data.lookup_value("currSec"),
                                ((env.data.lookup_value("currMiliSec") - 300) % 10000) / 100))
        
        obs= env.reset()
        sum_reward=0