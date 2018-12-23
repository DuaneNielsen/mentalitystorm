import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import imageio
import torch
import torch.utils
import torch.utils.data
from torch.utils import data as data_utils
from torchvision.transforms import functional as tvf
from mentalitystorm.data_containers import ObservationAction, DataSets, DataLoaders
from mentalitystorm.policies import RolloutGen
import numpy as np


class Selector(ABC):
    @abstractmethod
    def get_input(self, package, device):
        raise NotImplementedError

    @abstractmethod
    def get_target(self, package, device):
        raise NotImplementedError


class AutoEncodeSelect(Selector):
    def get_input(self, package, device):
        return package[0].to(device),

    def get_target(self, package, device):
        return package[0].to(device),


class StandardSelect(Selector):
    def __init__(self, source_index=0, target_index=1):
        self.source_index = source_index
        self.target_index = target_index

    def get_input(self, package, device):
        return package[self.source_index].to(device),

    def get_target(self, package, device):
        return package[self.target_index].to(device),


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


def collate_action_observation(batch):
    """
    collation function for ActionEncoderDataset
    :param batch: the batch
    :return: a minibatch
    """
    # short longest to shortest
    batch.sort(key=lambda x: x[2].shape[0], reverse=True)
    minibatch = [list(t) for t in zip(*batch)]
    return minibatch


def combine_action_latent(action, latent):
    return [torch.cat(act_z, dim=1)[:-1] for act_z in zip(action, latent)]


class ActionEncoderDataset(torch.utils.data.Dataset):
    """
    A dataset of observations, actions, rewards, and done
    two additional slots are provided
    screen - unfortunately the ai-gym atari observation space is sometimes different to the pixels returned
    latent - if an encoder model for the observations space is available, then the encoded
    space can be added here
    """
    def __init__(self, directory, load_screen=True, load_observation=True):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.count = 0
        self.load_screen = load_screen
        self.load_observation = load_observation
        for _ in self.path.glob('*.np'):
            self.count += 1

    def __getitem__(self, index):

        np_filepath = self.path / str(index)
        try:
            oa = ObservationAction.load(np_filepath.absolute(), load_observation=self.load_observation,
                                        load_screen=self.load_screen)
            framel = []
            if len(oa.screen) != 0:
                for frame in oa.screen:
                    framel.append(tvf.to_tensor(frame))
                screen = torch.stack(framel, dim=0)
            else:
                screen = None
        except Exception:
            print(f'error while loading {str(np_filepath)}')
            raise

        return screen, torch.Tensor(oa.observation), torch.Tensor(oa.action), \
               torch.Tensor(oa.reward), torch.Tensor(oa.done), torch.Tensor(oa.latent)

    def __len__(self):
        return self.count


class GymSimulatorDataset(torch.utils.data.Dataset):
    def __init__(self, env, policy, length, action_embedding, output_in_numpy_format=False):
        torch.utils.data.Dataset.__init__(self)
        self.length = length
        self.count = 0
        self.policy = policy
        self.rollout = RolloutGen(env, policy).__iter__()
        self.embed_action = action_embedding
        self.output_in_numpy_format = output_in_numpy_format

    def __getitem__(self, index):

        screen, observation, reward, done, info, action = self.rollout.next()
        latent = np.zeros(1)

        if not self.output_in_numpy_format:
            screen = tvf.to_tensor(screen)
            observation = torch.Tensor(observation)
            reward = torch.Tensor([reward])
            done = torch.Tensor([done])
            action = self.embed_action.tensor(action)
            latent = torch.Tensor(latent)

        self.count += 1

        return screen, observation, action, reward, done, latent

    def __len__(self):
        return self.length


class Splitter:

    def split(self, dataset):
        """
        Splits the dataset into dev, train and test
        :param dataset: the dataset to split
        :return: DataSets named tupple (dev, train, test)
        """
        dev = data_utils.Subset(dataset, range(len(dataset) * 2 // 10))
        train = data_utils.Subset(dataset, range(0, len(dataset) * 9//10))
        test = data_utils.Subset(dataset, range(len(dataset) * 9 // 10 + 1, len(dataset)))
        return DataSets(dev=dev, train=train, test=test)

    def loaders(self, dataset, **kwargs):
        """ Returns a named tuple of loaders

        :param kwargs: same as kwargs for torch.utils.data.Loader
        :return: named tuple of dataloaders, dev, train, test
        """

        split = self.split(dataset)

        dev = data_utils.DataLoader(split.dev, **kwargs)
        train = data_utils.DataLoader(split.train, **kwargs)
        test = data_utils.DataLoader(split.test, **kwargs)

        return DataLoaders(dev=dev, train=train, test=test)


class DataPackage:
    """ Datapackage provides everything required to load data to train and test a model
    """
    def __init__(self, dataset, selector, splitter=None):
        """
        :param dataset: the torch dataset object
        :param selector: selects the model input and model target data
        :param splitter: splits data into dev, train and test sets
        """
        self.dataset = dataset
        self.splitter = splitter if splitter is not None else Splitter()
        self.selector = selector

    def loaders(self, **kwargs):
        dev, train, test = self.splitter.loaders(self.dataset, **kwargs)
        return dev, train, test, self.selector