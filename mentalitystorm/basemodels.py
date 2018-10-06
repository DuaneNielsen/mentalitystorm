import torch
from torch import nn as nn

from mentalitystorm import Dispatcher, Observable, Trainable, TensorBoardObservable


class BaseAE(nn.Module):
    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class MultiChannelAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.ae_l = []
        self.ch_l = []

    def add_ae(self, auto_encoder, channels):
        """

        :param auto_encoder: the BaseAE to use
        :param channels: LongTensor of channels to use for input
        """
        self.ae_l.append(auto_encoder)
        self.ch_l.append(channels)

    def forward(self, x):
        z_l = []
        for channels, ae in zip(self.ch_l, self.ae_l):
            z = ae.encode(x[:, channels, :, :])
            z_l.append(z)
        result = torch.cat(z_l, dim=1)
        return result


class DummyCoder(nn.Module):
    def forward(self, x):
        return x


class DummyAE(BaseAE):
    def __init__(self):
        BaseAE.__init__(self, DummyCoder(), DummyCoder())


class BaseVAE(nn.Module, Dispatcher, Trainable, Observable, TensorBoardObservable):
    def __init__(self, encoder, decoder, variational=False):
        nn.Module.__init__(self)
        Dispatcher.__init__(self)
        TensorBoardObservable.__init__(self)

        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational

    def forward(self, x):
        indices = None
        logvar = None

        encoded = self.encoder(x)
        mu = encoded[0]
        logvar = encoded[1]
        if len(encoded) > 2:
            indices = encoded[2]

        self.metadata['z_size'] = mu[0].data.numel()

        z = self.reparameterize(mu, logvar)

        if indices is not None:
            decoded = self.decoder(z, indices)
        else:
            decoded = self.decoder(z)

        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training and self.variational:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, eps):
        images = self.decoder(eps)
        self.updateObserversWithImage('sample_image', images[0].data, training=self.training, always_write=True)
        return images
