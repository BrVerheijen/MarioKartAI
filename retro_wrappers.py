from collections import deque
import cv2
cv2.ocl.setUseOpenCL(False)
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv, MaxAndSkipEnv
from utils import TimeLimit
import numpy as np
import gym

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, action):
        done = False
        total_reward = 0
        for i in range(self.n):

            if self.curac is None:
                self.curac = action
            
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = action

            elif i==1:
                self.curac = action

            if self.supports_want_render and i< self.n-1:
                obs, reward, done, info = self.env.step(self.curac, want_render=False)
            else:
                obs, reward, done, info = self.env.step(self.curac)

            total_reward += reward
            if done: break
        return obs, total_reward, done, info

    def seed(self, s):
        self.rng.seed(s)

class PartialFrameStack(gym.Wrapper):
    def __init__(self, env, k, channel=1):
        """
        Stack one channel (channel keyword) from previous frames
        """
        gym.Wrapper.__init__(self, env)
        _shape = env.observation_space.shape
        self.channel = channel
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(_shape[0], _shape[1], _shape[2] + k - 1),
            dtype= env.observation_space.dtype)
        self.k = k
        self.frames = deque([], maxlen=k)
        _shape = env.observation_space.shape

    def reset(self):
        obs = self.env.reset()
        assert obs.shape[2] > self.channel
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate([frame if i==self.k-1 else frame[:,:,self.channel:self.channel+1]
            for (i, frame) in enumerate(self.frames)], axis=2)


class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio):
        """
        Downsample images by a factor of ratio
        """
        #Observation space shape onderzoeken !!!!!!!!!!!!
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh//ratio, oldw//ratio, oldc//ratio)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame


class Rgb2gray(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Downsample images by a factor of ratio
        """
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(oldh, oldw, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:,:,None]


class MovieRecord(gym.Wrapper):
    def __init__(self, env, savedir, k):
        gym.Wrapper.__init__(self, env)
        self.savedir = savedir
        self.k = k
        self.episode_count = 0
    
    def reset(self):
        if self.episode_count % self.k == 0:
            self.env.unwrapped.movie_path = self.savedir
        else:
            self.env.unwrapped.movie_path = None
            self.env.unwrapped.movie = None
        self.episode_count += 1
        return self.env.reset()


class AppendTimeout(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = env.action_space
        self.timeout_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        self.original_os = env.observation_space

        if isinstance(self.original_os, gym.spaces.Dict):
            import copy
            ordered_dict = copy.deepcopy(self.original_os.spaces)
            ordered_dict['value_estimation_timeout'] = self.timeout_space
            self.observation_space = gym.spaces.Dict(ordered_dict)
            self.dict_mode = True

        else:
            self.observation_space = gym.spaces.Dict({
                'original': self.original_os,
                'value_estimation_timeout': self.timeout_space
            })
            self.dict_mode= False
        self.action_count = None

        while 1:
            if not hasattr(env, "_max_episode_steps"):
                env = env.env
                continue
            break
        self.timeout = env._max_episode_steps


    def step(self, action):
        self.action_count += 1
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        self.action_count = 0
        return self._process(self.env.reset())

    def _process(self, obs):
        fracmissing = 1 -self.action_count / self.timeout
        if self.dict_mode:
            obs['value_estimation_timeout'] = fracmissing
        else:
            return { 'original': obs, 'value_estimation_timeout': fracmissing}


class StartDoingRandomActionsWrapper(gym.Wrapper):
    """
    Warning: can eat info dicts, not good if you depend on them
    """
    def __init__(self, env, max_random_steps, on_startup=True, every_episode=False):
        gym.Wrapper.__init__(self, env)
        self.on_startup = on_startup
        self.every_episode = every_episode
        self.random_steps = max_random_steps
        self.last_obs = None
        if on_startup:
            self.some_random_steps()

    def some_random_steps(self):
        self.last_obs = self.env.reset()
        n = np.random.randint(self.random_steps)

        for _ in range(n):
            self.last_obs, _, done, _ = self.env.step(self.env.action_space.sample())
            if done: self.last_obs = self.env.reset()

    def reset(self):
        return self.last_obs

    def step(self, action):
        self.last_obs, reward, done, info = self.env.step(action)
        if done:
            self.last_obs = self.env.reset()
            if self.every_episode:
                self.some_random_steps()
        return self.last_obs, reward, done, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    import retro
    if state is None:
        state = retro.State.DEFAULT

    env = retro.make(game, state, **kwargs)

    env = StochasticFrameSkip(env, n=4, stickprob=0.25)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env, scale=True, frame_stack=4):
    
    env = WarpFrame(env)
    env = ClipRewardEnv(env)

    return env


class RewardScaler(gym.Wrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class AllowBackTracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBackTracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._cur_x += reward
        reward = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, reward, done, info

