import pickle
from collections import namedtuple
from pathlib import Path
import imageio
import torch.utils.data as data_utils
imageio.plugins.ffmpeg.download()
import numpy as np
import torch
import torch.utils
from torchvision.transforms import functional as tvf
import torchvision.transforms as TVT
from intervaltree import IntervalTree
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

        a = self.action_embedding.numpy(action)
        #act_n = a.cpu().numpy()
        act_n = np.expand_dims(a, axis=0)

        if self.oa is None:
            self.oa = ObservationAction(filename=filename)

        self.oa.add(screen_n, observation, act_n, reward, done, latent_n)

    def save_session(self):
        self.oa.end_episode()
        self.oa.save()
        self.oa = None


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
        self.action = np.stack(self.action_l, axis=0)
        self.reward = np.array(self.reward_l, dtype='float32')
        self.done = np.array(self.done_l, dtype='float32')
        if len(self.observation_l) != 0:
            self.observation = np.stack(self.observation_l, axis=0)
        else:
            self.observation = []

        if len(self.latent_l) != 0:
            self.latent = np.stack(self.latent_l, axis=0)
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
    def load(filename):
        np_fn = str(filename) + '.np'
        mp4_fn = str(filename) + '.mp4'
        with open(np_fn, 'rb') as f:
            oa = pickle.load(f)
        if Path(mp4_fn).exists():
            reader = imageio.get_reader(mp4_fn)
            frames = [list(frame) for frame in reader]
            oa.screen = np.stack(frames, axis=0)
        else:
            oa.screen = []
        return oa


class ActionEmbedding():
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


class ActionEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.count = 0
        for _ in self.path.glob('*.np'):
            self.count += 1

    def __getitem__(self, index):
        np_filepath = self.path / str(index)
        oa = ObservationAction.load(np_filepath.absolute())
        framel = []
        for frame in oa.screen:
            framel.append(tvf.to_tensor(frame))
        screen = torch.stack(framel, dim=0)

        return screen, torch.Tensor(oa.observation), torch.Tensor(oa.action), \
               torch.Tensor(oa.reward), torch.Tensor(oa.done), torch.Tensor(oa.latent)

    def __len__(self):
        return self.count


class GymImageDataset(data_utils.Dataset):
    def __init__(self, directory, input_transform=None, target_transform=None):
        self.pngs = sorted(Path(directory).glob('pic*.png'))
        self.rewards = sorted(Path(directory).glob('rew*.np'))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        with self.pngs[index].open('rb') as f:
            image = imageio.imread(f)

            if self.input_transform is not None:
                input_image = self.input_transform(image)
            else:
                input_image = image

            if self.target_transform is not None:
                target_image = self.target_transform(image)
            else:
                target_image = image

        with self.rewards[index].open('rb') as f:
            reward = pickle.load(f)
        return input_image, target_image, reward

    def __len__(self):
        return len(self.pngs)

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


def collate_action_observation(batch):
    # short longest to shortest
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    minibatch = [list(t) for t in zip(*batch)]
    # first frame has ridiculous high variance, so drop it, I
    #clean = drop_first_frame(minibatch)
    #delta = observation_deltas(clean)
    return minibatch


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