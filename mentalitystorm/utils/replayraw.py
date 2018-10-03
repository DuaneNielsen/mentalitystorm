import torch
from mentalitystorm.atari import ActionEncoderDataset, collate_action_observation
import torch.utils.data as data_utils
from mentalitystorm import ImageViewer
import numpy as np

screen_viewer = ImageViewer('input', (320, 480))

dataset = ActionEncoderDataset(r'c:\data\SpaceInvaders-v4\rl_raw_v1')

loader = data_utils.DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True,
                               collate_fn=collate_action_observation)

for screen, observation, action, reward, done, latent in loader:
    for frame in screen[0]:
        screen_viewer.update(frame, 'tensor_gym_RGB')