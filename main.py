
import retro

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy

import os
import traceback

from mario_wrappers import *
from retro_wrappers import wrap_deepmind_retro
from utils import SaveOnBestTrainingRewardCallbackCustom

workers = 4
steps = 2000
log_dir = './ppo_test'

scenario= 'scenario'

# state = ["BowserCastle_M", "BowserCastle2_M", "BowserCastle3_M", "ChocoIsland_M", "ChocoIsland2_M", "DonutPlains_M",
#           "DonutPlains2_M", "DonutPlains3_M", "GhostValley_M", "GhostValley2_M", "GhostValley3_M", "KoopaBeach_M",
#           "KoopaBeach2_M", "MarioCircuit_M", "MarioCircuit2_M", "MarioCircuit3_M", "MarioCircuit4_M", "RainbowRoad_M",
#           "VanillaLake_M", "VanillaLake2_M"]
state= retro.State.DEFAULT

ALGORITHM = PPO

RENDER= True

def wrapper(env):
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    env = TimeLimitWrapperMarioKart(env, minutes= 3, seconds=0)
    env = CutMarioMap(env, show_map=False)
    env = wrap_deepmind_retro(env)
    return env

def main():

    env = retro_make_vec_env('SuperMarioKart-Snes', scenario=scenario, state=state, n_envs=workers,
                            monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, wrapper_class=wrapper, seed=0)


    callback = SaveOnBestTrainingRewardCallbackCustom( check_freq=1000, log_dir=log_dir)

    if os.path.exists(f"{log_dir}/best_model.zip"):
        print("LOAD BEST MODEL")
        model = ALGORITHM.load(f"{log_dir}/best_model.zip")
        model.set_env(env)

        model.verbose = 1
        model.tensorboard_log = log_dir

    else: 
        model = ALGORITHM(CnnPolicy, env, verbose=1, tensorboard_log=log_dir)

    try:
        model.learn(total_timesteps=steps, callback=callback)
        model.save(f"{log_dir}/model_backup")
        print("SAVE MODEL")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/model_backup")
        print("SAVED MODEL KEYBOARD INTERRUPT")
    except Exception as e:
        print("EXCEPTION",e)
        traceback.print_exc()

        model.save(f"{log_dir}/model_backup")
        print("SAVED MODEL exception")

    if RENDER:
        obs = env.reset()

        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(action, reward)
            env.render('human')

if __name__ == "__main__":
    main()