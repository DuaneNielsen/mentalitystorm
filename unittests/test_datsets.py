from unittest import TestCase
import gym
from mentalitystorm.config import config
from mentalitystorm.storage import Storeable
import torch
from mentalitystorm.policies import VCPolicy
from mentalitystorm.data import GymSimulatorDataset
import torch.utils.data

class TestGymSim(TestCase):
    def test_gymsim(self):
        env = gym.make('SpaceInvaders-v4')
        models = config.basepath() / 'SpaceInvaders-v4' / 'models'
        visualsfile = models / 'GM53H301W5YS38XH'
        visuals = Storeable.load(str(visualsfile)).to(config.device())
        controllerfile = models / 'best_model68'
        controller = torch.load(str(controllerfile))
        policy = VCPolicy(visuals, controller)


        dataset = GymSimulatorDataset(env, policy, 3000)

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=True)

        for screen, observation, action, reward, done, _ in loader:
            print(reward)
