import gymnasium as gym
import numpy as np
from IPython import embed

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.env = env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = np.logical_or(terminated, truncated)
        lives = self.env.unwrapped.env._life
        if self.lives > lives > 0:
            terminated, truncated = True, True
        self.lives = lives
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.env = env

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if np.logical_or(terminated, truncated):
                break
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        return self.env.reset()
    
    def render(self):
        return self.env.render()


class Gym2Gymnasium(gym.Wrapper):
    def __init__(self, env):
        """Convert gym.Env to gymnasium.Env"""
        self.env = env

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def step(self, action):
        """Repeat action, and sum reward"""
        return self.env.step(action)

    def reset(self, options=None, seed=None):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)