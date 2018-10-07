from unittest import TestCase
import gym
from mentalitystorm.config import config
from mentalitystorm.storage import Storeable
from mentalitystorm.policies import VCPolicy, RolloutGen
from mentalitystorm.observe import ImageViewer
import torch


class TestRolloutGen(TestCase):
    def test_rollout_gen(self):
        env = gym.make('SpaceInvaders-v4')
        models = config.basepath() / 'SpaceInvaders-v4' / 'models'
        visualsfile = models / 'GM53H301W5YS38XH'
        visuals = Storeable.load(str(visualsfile)).to(config.device())
        controllerfile = models / 'best_model68'
        controller = torch.load(str(controllerfile))
        policy = VCPolicy(visuals, controller)

        viewer = ImageViewer('screen', (420, 360), 'numpyRGB')

        for screen, observation, reward, done, info, action in RolloutGen(env, policy):
            viewer.update(screen)

