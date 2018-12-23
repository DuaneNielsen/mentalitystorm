import pickle
from collections.__init__ import namedtuple
from pathlib import Path

import imageio
import numpy as np
import torch

from mentalitystorm.image import NumpyRGBWrapper


class ObservationAction:
    def __init__(self, filename):

        self.observation_l = []
        self.action_l = []
        self.reward_l = []
        self.done_l = []
        self.latent_l = []

        self.screen = None
        self.observation = None
        self.action = None
        self.reward = None
        self.done = None
        self.latent = None

        self.mp4_path = str(filename) + '.mp4'
        Path(self.mp4_path).parent.mkdir(parents=True, exist_ok=True)
        self.video_writer = ImageVideoWriter(self.mp4_path)
        self.filename = filename

        self.length = 0

    def add(self, screen, observation, action, reward, done, latent=None):

        if screen is not None:
            self.video_writer.add_frame(screen, 'numpyRGB3')
        if observation is not None:
            self.observation_l.append(observation)
        if latent is not None:
            self.latent_l.append(latent)

        self.action_l.append(action)
        self.reward_l.append(reward)
        self.done_l.append(done)
        self.length += 1

    def __len__(self):
        return self.length

    def end_episode(self):
        self.action = np.stack(self.action_l, axis=0).squeeze()
        self.reward = np.array(self.reward_l, dtype='float32')
        self.done = np.array(self.done_l, dtype='float32')
        if len(self.observation_l) != 0:
            self.observation = np.stack(self.observation_l, axis=0)
        else:
            self.observation = []

        if len(self.latent_l) != 0:
            self.latent = np.stack(self.latent_l, axis=0).squeeze()
        else:
            self.latent = []

        self.observation_l.clear()
        self.action_l.clear()
        self.reward_l.clear()
        self.done_l.clear()
        self.latent_l.clear()
        self.video_writer.endSession()

    def save(self, filename=None):

        if filename is None:
            filename = self.filename

        from pathlib import Path
        path = Path(filename + '.np')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.absolute(), 'wb') as f:
            pickle.dump(self, file=f)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['screen'] = None
        state['video_writer'] = None
        return state

    @staticmethod
    def load(filename, load_observation=True, load_screen=True):
        np_fn = str(filename) + '.np'
        mp4_fn = str(filename) + '.mp4'
        with open(np_fn, 'rb') as f:
            oa = pickle.load(f)
        if not load_observation:
            oa.observation = []
        if Path(mp4_fn).exists() and load_screen:
            reader = imageio.get_reader(mp4_fn)
            frames = [list(frame) for frame in reader]
            oa.screen = np.stack(frames, axis=0)
        else:
            oa.screen = []
        return oa


class ImageVideoReader:
    def __init__(self, file):
        self.reader = imageio.get_reader(self.file)

    def play(self):
        for i, im in enumerate(self.reader):
            print('mean of frame %i is %1.1f' % (i, im.mean()))


class ImageVideoWriter:
    def __init__(self, file):
        self.writer = None
        self.file = file
        self.writer = imageio.get_writer(self.file, macro_block_size=None)

    def add_frame(self, screen, format):

        frame = NumpyRGBWrapper(screen, format).numpyRGB
        self.writer.append_data(screen)

    def endSession(self):
        self.writer.close()


RLStep = namedtuple('Step', 'screen, observation, action, reward, done, meta')


class ActionEmbedding:
    """
    Simple one-hot embedding of the action space
    """
    def __init__(self, env):
        self.env = env

    def tensor(self, action):
        action_t = torch.zeros(self.env.action_space.n)
        action_t[action] = 1.0
        return action_t

    def numpy(self, action):
        action_n = np.zeros(self.env.action_space.n)
        action_n[action] = 1.0
        return action_n

    def embedding_to_action(self, index):
        return index

    def start_tensor(self):
        return torch.zeros(self.env.action_space.n)

    def start_numpy(self):
        return np.zeros(self.env.action_space.n)


class ThreeKeyEmbedding:
    """
    Op      Policy    Environment
    Noop    0         0
    Right   1         3
    Left    2         4
    Fire    4         1
    """
    def __init__(self):
        self.to_env = np.array([0, 3, 4, 1, -1])
        self.to_policy = self.get_reverse_lookup(self.to_env)

    def get_reverse_lookup(self, a):
        """
        computes a permutation matrix for a lookup table
        then inverts it to get the reverse lookup !
        :param a: the lookup table to invert
        :return: the reverse lookup table
        """
        N = a.size
        rows = np.arange(N)
        P = np.zeros((N, N), dtype=int)
        P[rows, a] = 1
        return np.where(P.T)[1]

    def toPolicy(self, action):
        return self.to_policy[action]

    def toEnv(self, index):
        return self.to_env[index]

    def tensor(self, action):
        action_t = torch.zeros(4)
        action = self.to_policy[action]
        action_t[action] = 1.0
        return action_t

    def numpy(self, action):
        action_n = np.zeros(self.env.action_space.n)
        action = self.to_policy[action]
        action_n[action] = 1.0
        return action_n





DataSets = namedtuple('DataSets', 'dev, train, test')
DataLoaders = namedtuple('DataSets', 'dev, train, test')