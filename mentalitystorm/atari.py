from pathlib import Path
import imageio
import torch.utils.data as data_utils

from mentalitystorm.data_containers import ObservationAction, ActionEmbedding

imageio.plugins.ffmpeg.download()
import numpy as np
import torch
import torch.utils
from torchvision.transforms import functional as tvf
from intervaltree import IntervalTree


class ActionEncoder:
    """
    update takes a tuple of ( numpyRGB, integer )
    and saves as tensors
    """
    def __init__(self, env, name, action_embedding, model=None):
        self.model = model
        if model is not None:
            self.model.eval()
        self.session = 1
        self.sess_obs_act = None
        self.action_embedding = action_embedding
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
                mu, logsigma = self.model.encoder(screen_t.to(self.device))
                latent_n = mu.detach().cpu().numpy()

        if action is not None:
            a = self.action_embedding.numpy(action)
        else:
            a = self.action_embedding.start_numpy()

        act_n = np.expand_dims(a, axis=0)

        if self.oa is None:
            self.oa = ObservationAction(filename=filename)

        self.oa.add(screen_n, observation, act_n, reward, done, latent_n)

    def save_session(self):
        self.oa.end_episode()
        self.oa.save()
        self.oa = None


class CachePage:
    def __init__(self, path):
        self.index = path.name
        oa = ObservationAction.load(path)
        self.length = len(oa)

    def __len__(self):
        return self.length


class FrameByFrameDataset(torch.utils.data.Dataset):
    """
    INCOMPLETE
    """
    def __init__(self, directory):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.pagemap = IntervalTree()
        self.count = 0
        for _ in self.path.iterdir():
            self.count += 1
        self.buildCacheMap()

    def buildCacheMap(self):
        cursor = 0
        for file_id in range(self.count):
            path = self.directory / file_id
            page = CachePage(path)
            self.pagemap[cursor, cursor + len(page) - 1] = page
            cursor += len(page)

    def __len__(self):
        return self.count


def drop_first_frame(minibatch):
    latent_l = []
    action_l = []
    for latent in minibatch[0]:
        latent_l.append(latent[1:, :])
    for action in minibatch[1]:
        action_l.append(action[1:, :])
    return latent_l, action_l


def observation_deltas(minibatch):
    latent_l = []
    latent_raw_l = []
    action_l = []
    first_frame_l = []

    for first_frame in minibatch[0]:
        first_frame_l.append(first_frame[0, :])

    for latent in minibatch[0]:
        latent_l.append(latent[1:, :] - latent[:-1, :])

    for action in minibatch[1]:
        action_l.append(action[:-1, :])

    for latent in minibatch[0]:
        latent_raw_l.append(latent[1:, :])

    return latent_raw_l, first_frame_l, latent_l, action_l