import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CueDelayChoiceEnv(gym.Env):
    def __init__(self, delay=10):
        super().__init__()
        self.delay = int(delay)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
        self.max_steps = self.delay + 2

        self._t = 0
        self._cue = 0

    def _obs(self, phase, show_cue: bool):
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        obs[0, 1, 0] = np.uint8(phase)
        if show_cue:
            obs[0, 0, 0] = np.uint8(self._cue + 1)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._cue = int(self.np_random.integers(0, 2))
        obs = self._obs(phase=1, show_cue=True)
        info = {}
        return obs, info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0

        if self._t == 0:
            self._t += 1
            obs = self._obs(phase=2 if self.delay > 0 else 3, show_cue=False)
            return obs, reward, terminated, truncated, {}

        if 1 <= self._t <= self.delay:
            self._t += 1
            phase = 2 if self._t <= self.delay else 3
            obs = self._obs(phase=phase, show_cue=False)
            return obs, reward, terminated, truncated, {}

        if int(action) == self._cue:
            reward = 1.0
        terminated = True
        self._t += 1
        obs = self._obs(phase=3, show_cue=False)
        return obs, reward, terminated, truncated, {}
