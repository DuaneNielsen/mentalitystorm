import pickle
from collections.__init__ import namedtuple
from pathlib import Path

import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
import torch
import torch.utils
from torchvision.transforms import functional as tvf

from .image import NumpyRGBWrapper


RLStep = namedtuple('Step', 'screen, observation, action, reward, done, meta')


class ActionEncoder:
    """
    update takes a tuple of ( numpyRGB, integer )
    and saves as tensors
    """
    def __init__(self, env, name, model=None):
        self.model = model
        if model is not None:
            self.model.eval()
        self.session = 1
        self.sess_obs_act = None
        self.action_embedding = ActionEmbedding(env)
        self.device = torch.device('cpu')
        self.name = name
        self.env = env
        self.oa = None

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        self.device = device
        return self

    def update(self, rlstep):
        observation = rlstep.observation
        action = rlstep.action
        reward = rlstep.reward
        done = rlstep.done
        latent_n = None
        screen_n = None
        filename = rlstep.meta['filename']

        if rlstep.screen is not None:
            screen_t = tvf.to_tensor(rlstep.screen.copy()).detach().unsqueeze(0).to(self.device)
            screen_n = rlstep.screen

        """ if a model is set, use it to compress the screen"""
        if self.model is not None:
            with torch.no_grad():
                self.model.eval()
            mu, logsigma = self.model.encoder(screen_t)
            latent_n = mu.detach().cpu().numpy()

        a = self.action_embedding.toNumpy(action)
        #act_n = a.cpu().numpy()
        act_n = np.expand_dims(a, axis=0)

        if self.oa is None:
            self.oa = ObservationAction(filename=filename)

        self.oa.add(screen_n, observation, act_n, reward, done, latent_n)

    def save_session(self):
        self.oa.save()
        self.oa = None


class ObservationAction:
    def __init__(self, filename):
        mp4_path = str(filename) + '.mp4'
        self.screen = ImageVideoWriter(mp4_path)
        self.observation = []
        self.action = []
        self.reward = []
        self.done = []
        self.latent = []
        self.filename = filename

    def add(self, screen, observation, action, reward, done, latent=None):
        if screen is not None:
            self.screen.add_frame(screen, 'numpyRGB3')
        if observation is not None:
            self.observation.append(observation)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        if latent is not None:
            self.latent.append(latent)

    def save(self):
        self.screen.endSession()
        if len(self.observation) != 0:
            self.observation = np.stack(self.observation, axis=0)
        self.action = np.array(self.action, dtype='float32')
        self.reward = np.array(self.reward, dtype='float32')
        self.done = np.array(self.done, dtype='float32')
        if len(self.latent) != 0:
            self.latent = np.stack(self.latent, axis=0)

        from pathlib import Path
        path = Path(self.filename + '.np')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.absolute(), 'wb') as f:
            pickle.dump(self, file=f)

    @staticmethod
    def convert_frame(frame):
        return tvf.to_tensor(frame)

    @staticmethod
    def load(filename, image_transforms=None):
        mp4_fn = str(filename) + '.mp4'
        np_fn = str(filename) + '.np'
        with open(np_fn, 'rb') as f:
            oa = pickle.load(f)
            if Path(mp4_fn).exists():
                reader = imageio.get_reader(mp4_fn)
                frames = []
                for frame in reader:
                    frame = ObservationAction.convert_frame(frame)
                    frames.append(frame)
                oa.screen = torch.stack(frames, dim=0)
        return oa



class ActionEmbedding():
    def __init__(self, env):
        self.env = env

    def toTensor(self, action):
        action_t = torch.zeros(self.env.action_space.n)
        action_t[action] = 1.0
        return action_t

    def toNumpy(self, action):
        action_n = np.zeros(self.env.action_space.n)
        action_n[action] = 1.0
        return action_n


class ActionEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.count = 0
        for _ in self.path.iterdir():
            self.count += 1

    def __getitem__(self, index):
        np_filepath = self.path / str(index)
        oa = ObservationAction.load(np_filepath.absolute())

        return torch.Tensor(oa.screen), torch.Tensor(oa.observation), torch.Tensor(oa.action), \
               torch.Tensor(oa.reward), torch.Tensor(oa.done), torch.Tensor(oa.latent)

    def __len__(self):
        return self.count


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

