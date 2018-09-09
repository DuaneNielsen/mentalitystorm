import torch
from torch import nn as nn

from mentalitystorm import Dispatcher, Observable, Trainable, TensorBoardObservable


class BaseVAE(nn.Module, Dispatcher, Observable, Trainable, TensorBoardObservable):
    def __init__(self, encoder, decoder, variational=False):
        nn.Module.__init__(self)
        Dispatcher.__init__(self)
        TensorBoardObservable.__init__(self)

        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational

    def forward(self, x):
        input_shape = x.shape
        indices = None

        self.updateObserversWithImage('input', x[0], training=self.training)

        encoded = self.encoder(x)
        mu = encoded[0]
        logvar = encoded[1]
        if len(encoded) > 2:
            indices = encoded[2]

        # if z can be shown as an image dispatch it
        if mu.shape[1] == 3 or mu.shape[1] == 1:
            self.updateObserversWithImage('z', mu[0].data)
        self.metadata['z_size'] = mu[0].data.numel()

        z = self.reparameterize(mu, logvar)

        if indices is not None:
            decoded = self.decoder(z, indices)
        else:
            decoded = self.decoder(z)

        # should probably make decoder return same shape as encoder
        decoded = decoded.view(input_shape)

        self.updateObserversWithImage('output', decoded[0].data, training=self.training)

        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training and self.variational:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
