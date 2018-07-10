import gym
import numpy as np
from gym.spaces import Box, Discrete
from gym import ActionWrapper, ObservationWrapper

import cv2


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 42, 42])
    return frame


class AtariRescale42x42(ObservationWrapper):

    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


class NormalizedEnv(ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]

class Vectorize(gym.Wrapper):
    """
Given an unvectorized environment (where, e.g., the output of .step() is an observation
rather than a list of observations), turn it into a vectorized environment with a batch of size
1.
"""

    def __init__(self, env):
        super(Vectorize, self).__init__(env)
        self.metadata = {'runtime.vectorized': True}
        assert not env.metadata.get('runtime.vectorized')
        assert self.metadata.get('runtime.vectorized')
        self.n = 1

    def _reset(self):
        observation = self.env.reset()
        return [observation]

    def _step(self, action):
        observation, reward, done, info = self.env.step(action[0])
        return [observation], [reward], [done], {'n': [info]}

    def _seed(self, seed):
        return [self.env.seed(seed[0])]

class Env(gym.Env):
    """Base class capable of handling vectorized environments.
    """
    metadata = {
        # This key indicates whether an env is vectorized (or, in the case of
        # Wrappers where autovectorize=True, whether they should automatically
        # be wrapped by a Vectorize wrapper.)
        'runtime.vectorized': True,
    }

    # Number of remotes. User should set this.
    n = None

class Wrapper(Env, gym.Wrapper):
    """Use this instead of gym.Wrapper iff you're wrapping a vectorized env,
    (or a vanilla env you wish to be vectorized).
    """
    # If True and this is instantiated with a non-vectorized environment,
    # automatically wrap it with the Vectorize wrapper.
    autovectorize = True

    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        if not env.metadata.get('runtime.vectorized'):
            if self.autovectorize:
                # Circular dependency :(
                env = Vectorize(env)
            else:
                raise Exception('This wrapper can only wrap vectorized envs (i.e. where env.metadata["runtime.vectorized"] = True), not {}. Set "self.autovectorize = True" to automatically add a Vectorize wrapper.'.format(env))

        self.env = env

    @property
    def n(self):
        return self.env.n

    def configure(self, **kwargs):
        self.env.configure(**kwargs)


class Unvectorize(Wrapper):
    """
Take a vectorized environment with a batch of size 1 and turn it into an unvectorized environment.
"""
    autovectorize = False
    metadata = {'runtime.vectorized': False}

    def _reset(self):
        observation_n = self.env.reset()
        assert(len(observation_n) == 1)
        return observation_n[0]

    def _step(self, action):
        action_n = [action]
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        return observation_n[0], reward_n[0], done_n[0], info['n'][0]

    def _seed(self, seed):
        return self.env.seed([seed])[0]


