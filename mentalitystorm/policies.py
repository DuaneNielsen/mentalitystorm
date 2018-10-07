from abc import ABC, abstractmethod

import torch
from torchvision.transforms import ToTensor

from mentalitystorm import Dispatcher, Observable, Hookable, RLStep


class Policy(ABC):
    @abstractmethod
    def action(self, screen, observation): raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, env):
        self.env = env

    def action(self, screen, observation):
        return self.env.action_space.sample()


class VCPolicy(Policy):
    def __init__(self, v, c):
        self.v = v.eval()
        self.c = c.eval()
        self.trans = ToTensor()

    def action(self, screen, observation):
        obs = torch.tensor(screen).cuda().float()
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        latent, sigma = self.v.encoder(obs)
        latent = latent.cpu().double().squeeze(3).squeeze(2)
        action = self.c(latent)
        _, the_action = action.max(1)
        return the_action.item()


class Rollout(Dispatcher, Observable, Hookable):
    def __init__(self, env):
        Hookable.__init__(self)
        self.env = env

    def register_end_session(self, hook):
        self.register_after_hook(hook)

    def end_session(self, session):
        self.execute_after(session)

    def register_step(self, hook):
        self.register_before_hook(hook)

    def execute_step_hook(self, step):
        self.execute_before(step)

    def rollout(self, policy, episode, max_timesteps=100):
        observation = self.env.reset()
        screen = self.env.render(mode='rgb_array')
        reward = 0
        done = False
        action = policy.action(screen, observation)
        meta = {'episode': episode}
        self.execute_step_hook(RLStep(screen, observation, action, reward, done, meta))

        for t in range(max_timesteps):

            observation, reward, done, info = self.env.step(action)
            screen = self.env.render(mode='rgb_array')
            action = policy.action(screen, observation)

            self.execute_step_hook(RLStep(screen, None, action, reward, done, meta))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        self.end_session(episode)