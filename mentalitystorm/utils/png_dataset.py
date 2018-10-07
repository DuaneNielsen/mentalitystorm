import torch
from mentalitystorm.atari import ActionEncoderDataset, collate_action_observation
import torch.utils.data as data_utils
from mentalitystorm import ImageViewer, NumpyRGBWrapper
import imageio
from pathlib import Path
import pickle


def main(dataset_size, action_encoder_dir, output_dir):

    outputdir = Path(output_dir)
    outputdir.mkdir(parents=True, exist_ok=True)

    screen_viewer = ImageViewer('input', (320, 480), format='tensor_gym_RGB')

    dataset = ActionEncoderDataset(action_encoder_dir)

    loader = data_utils.DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True,
                                   collate_fn=collate_action_observation)

    num_images = 0

    for episode, (screen, observation, action, reward, done, latent) in enumerate(loader):

        if num_images >= dataset_size:
            break

        step = 10
        length = screen[0].shape[0]
        length = length - length % step
        index = torch.Tensor(range(0, length, step)).long()
        offsets = torch.randint_like(index, 0, step-1)
        index = index + offsets

        sampled_screen = screen[0][index]
        sampled_reward = reward[0][index]

        for i, (frame, reward) in enumerate(zip(sampled_screen, sampled_reward)):

            if num_images >= dataset_size:
                break

            w_frame = NumpyRGBWrapper(frame, 'tensor_gym_RGB')
            file = outputdir / Path('pic%04d_%04d' % (episode, i)).with_suffix('.png')
            imageio.imwrite(file, w_frame.numpyRGB)
            file = outputdir / Path('rew%04d_%04d' % (episode, i)).with_suffix('.np')
            with file.open(mode='wb') as f:
                pickle.dump(reward, f)
            screen_viewer.update(frame)
            num_images += 1
            print(num_images)


if __name__ == '__main__':
    main(80000, r'C:\data\SpaceInvaders-v4\rl_raw_v1',r'c:\data\SpaceInvaders-v4\images\raw_v2\all')