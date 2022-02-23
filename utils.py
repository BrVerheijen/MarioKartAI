import gym
from gym import spaces

import numpy as np
import os 

from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import BaseCallback



class SaveOnBestTrainingRewardCallBack(BaseCallback):
    """
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq: (int)
	:param log_dir: (str) Path to the folder where the model will be saved.
	  It must contains the file created by the ``Monitor`` wrapper.
	:param verbose: (int)
	"""

    def __init__(self, check_freq: int, log_dir: str, verbose= 1):
        super(SaveOnBestTrainingRewardCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            x,y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:

                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}"
                        .format(self.best_mean_reward, mean_reward)
                    )

                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    if self.verbose > 0:
                        print("Saving new best model {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class SaveOnBestTrainingRewardCallbackCustom(BaseCallback):
    """
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq: (int)
	:param log_dir: (str) Path to the folder where the model will be saved.
	  It must contains the file created by the ``Monitor`` wrapper.
	:param verbose: (int)
	"""

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallbackCustom, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        self.auto_saves_timesteps = [
			100_000, 150_000, 200_000, 250_000, 400_000, 500_000, 750_000,
			1_000_000, 1_500_000, 2_000_00, 2_500_000, 3_000_000, 3_500_000,
			4_000_000, 5_000_000, 7_000_000, 10_000_000
		]

        self.auto_saves_timesteps.sort()

    def _init_callback(self) -> None:

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            x, y = ts2xy(load_results(self.log_dir), 'timesteps')

            mean_reward = np.mean(y[-100:])
            std_reward = np.std(y[-100:])
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print(
						"Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} +- {:.2f}"
						.format(self.best_mean_reward, mean_reward,
								std_reward))
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

                if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))

                self.model.save(self.save_path)

            if self.auto_saves_timesteps and self.num_timesteps>= self.auto_saves_timesteps[0]:

                periodic_save_path = os.path.join(self.log_dir, 'model_{}'.format(self.auto_saves_timesteps[0]))

                if self.verbose > 0:
                    print("Saving periodic model - {}".format(periodic_save_path))
                self.model.save(periodic_save_path)

                del self.auto_saves_timesteps[0]

        return True


class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps

        self.current_step = 0

    def reset(self):
    # Reset the environment
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
    
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)

        if self.current_step >= self.max_steps:
            done = True

            info['time_limit_reached'] = True
        return obs, reward, done, info

#Baselines common Wrappers

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated']=True
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrappers(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)