#Adds CUDA to dll directory, necessary for using GPU
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import retro

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy
from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import CnnPolicy as A2C_Policy

import traceback

from mario_wrappers import *
from retro_wrappers import wrap_deepmind_retro
from utils import SaveOnBestTrainingRewardCallbackCustom

#Main training settings
"""
workers: defines the amount of workers during training
steps: defines the amount of timesteps the AI should run for (timesteps are always rounded up tot the nearest multiplication of 2^13)
log_dir: defines the directory where files regarding the model and the logs
scenario: defines the scenario openAI retro should use (scenarios should consist of atleast a done function and a reward function)
ALGORITHM: defines which stablebaselines-3 algorithm should be used to train the AI/model
RENDER: defines whether or not the game should be rendered when training (rendering also prints the action and reward to the console)
state: defines the game state which openAI retro should use
"""
workers = 4
steps = 1000000
log_dir = './ppo_test'
scenario= 'scenario'
ALGORITHM = PPO
RENDER=False
state= retro.State.DEFAULT

# state = ["BowserCastle_M", "BowserCastle2_M", "BowserCastle3_M", "ChocoIsland_M", "ChocoIsland2_M", "DonutPlains_M",
#           "DonutPlains2_M", "DonutPlains3_M", "GhostValley_M", "GhostValley2_M", "GhostValley3_M", "KoopaBeach_M",
#           "KoopaBeach2_M", "MarioCircuit_M", "MarioCircuit2_M", "MarioCircuit3_M", "MarioCircuit4_M", "RainbowRoad_M",
#           "VanillaLake_M", "VanillaLake2_M"]


#Creates a wrapper for the environment
#Discretizer: creates actions for the AI to use in the environment
#TimeLimit: creates a timelimit for each episode executed in the environment
def wrapper(env):
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    env = TimeLimitWrapperMarioKart(env, minutes= 3, seconds=0)
    env = CutMarioMap(env, show_map=False)
    env = wrap_deepmind_retro(env)
    return env


def main():
    #sets up retro environment
    env = retro_make_vec_env('SuperMarioKart-Snes', scenario=scenario, state=state, n_envs=workers,
                            monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, wrapper_class=wrapper, seed=0)

    #saves model with highest mean reward
    callback = SaveOnBestTrainingRewardCallbackCustom( check_freq=100, log_dir=log_dir)

    #Checks whether or not the "best_model.zip" file exists, if it does it loads the model within it. If it creates a new model
    if os.path.exists(f"{log_dir}/best_model.zip"):
        print("LOAD BEST MODEL")
        model = ALGORITHM.load(f"{log_dir}/best_model.zip")
        model.set_env(env)

        model.device = "cuda"
        model.verbose = 1
        model.tensorboard_log = log_dir

    else: 
        model = ALGORITHM(CnnPolicy, env, verbose=1, tensorboard_log=log_dir, device="cuda")

    #Constantly keeps learning as long as the timesteps defined have not been reached, there is no keyboard interruption or exeption
    #Also saves model when stopped due to a keyboard interruption or exception
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

    #Defines the rendering of the environment when training
    if RENDER:
        obs = env.reset()

        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(action, reward)
            env.render('human')

if __name__ == "__main__":
    main()