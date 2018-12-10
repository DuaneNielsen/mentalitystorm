from abc import ABC, abstractmethod

import torch
from torchvision.transforms import ToTensor

from mentalitystorm.data_containers import RLStep, ActionEmbedding
from mentalitystorm.util import Hookable


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


class VCPolicyMultiAE(Policy):
    def __init__(self, v, c, t, action_embedding, device):
        self.v = v.eval()
        self.c = c.eval()
        self.trans = t
        self.device = device
        self.action_embedding = action_embedding

    def action(self, screen, observation):

        obs = self.trans(observation)
        obs = obs.unsqueeze(0).to(self.device)
        latent = self.v(obs)

        latent = latent.cpu().double().squeeze(3).squeeze(2)
        action = self.c(latent)
        _, the_action = action.max(1)
        return self.action_embedding.embedding_to_action(the_action.item())


class Rollout(Hookable):
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
        action = None
        meta = {'episode': episode}
        self.execute_step_hook(RLStep(screen, observation, action, reward, done, meta))

        for t in range(max_timesteps):

            action = policy.action(screen, observation)
            observation, reward, done, info = self.env.step(action)
            screen = self.env.render(mode='rgb_array')

            self.execute_step_hook(RLStep(screen, observation, action, reward, done, meta))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        self.end_session(episode)


class RolloutGen(object):
    def __init__(self, env, policy, populate_screen=True, render_to_window=False):
        self.env = env
        self.policy = policy
        self.done = True
        self.action = None
        self.populate_screen = populate_screen
        self.render_to_window = render_to_window

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def render(self):
        screen = None
        if self.populate_screen:
            screen = self.env.render(mode='rgb_array')
        if self.render_to_window:
            self.env.render()
        return screen

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        screen = self.render()
        action = self.policy.action(screen, observation)
        return screen, observation, reward, done, info, action

    def next(self):

        if self.done:
            observation = self.env.reset()
            screen = self.render()
            reward = 0
            self.done = False
            self.action = self.policy.action(screen, observation)
            info = {}
            return screen, observation, reward, self.done, info, self.action

        else:
            screen, observation, reward, done, info, action = self.step(self.action)
            self.action = action
            self.done = done
            return screen, observation, reward, done, info, action


