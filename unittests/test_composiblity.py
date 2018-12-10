from unittest import TestCase
import torch
import torch.nn as nn
from mentalitystorm.compose import CategoricalEncoder, PassThroughWrapper, FlattenInputWrapper, ReshapeOutputWrapper
from .models import LinearGaussianVAE
from mentalitystorm.grad_check import Checker


class TestComposer(TestCase):
    def test_channels(self):
        x_dog = torch.Tensor(10, 1, 10, 10)
        x_cat = torch.Tensor(10, 1, 10, 10)

        n = nn.Conv2d(1, 1, 2, 2)
        p = nn.Conv2d(1, 1, 3, 2)

        in_dict = {'cat': x_cat, 'dog': x_dog}

        m = CategoricalEncoder()
        m.categories['cat'] = n
        m.categories['dog'] = p

        y = m(in_dict)

        print(y['cat'].shape)
        print(y['dog'].shape)

    def test_sanity(self):
        from mentalitystorm.grad_check import Checker
        m = nn.Linear(2,2)
        Checker(m).grad_check(Checker.build_input((2, 2)))

    def test_passthrough(self):

        x = torch.Tensor(10, 2).double()
        encoder = LinearGaussianVAE.Encoder(2, 1).double()

        decoder = LinearGaussianVAE.Decoder(2, 1).double()

        Checker(decoder).grad_check(Checker.build_input((1, ), batch=True))

        decoder = PassThroughWrapper(decoder)

        autoencoder = nn.Sequential(encoder, decoder).double()

        y, mu, sigma = autoencoder(x)

        print(y)
        print(mu)
        print(sigma)

    def test_flatten_and_reshape(self):

        x = torch.Tensor(10, 1, 4, 4)

        conv_encoder = nn.Conv2d(1, 1, 2, 2)
        vae_encoder = LinearGaussianVAE.Encoder(4, 1)
        vae_encoder = FlattenInputWrapper(vae_encoder)
        encoder = nn.Sequential(conv_encoder, vae_encoder)

        z, mu, sigma = encoder(x)

        conv_decoder = nn.ConvTranspose2d(1, 1, 2, 2)
        vae_decoder = LinearGaussianVAE.Decoder(4, 1)
        vae_decoder = ReshapeOutputWrapper(vae_decoder, (1, 2, 2))
        decoder = nn.Sequential(vae_decoder, conv_decoder)

        y = decoder(z)

        assert y.shape == (10, 1, 4, 4)

    def test_flatten_and_reshape_with_passthrough(self):

        x = torch.randn(10, 1, 4, 4)

        conv_encoder = nn.Conv2d(1, 1, 2, 2)
        vae_encoder = LinearGaussianVAE.Encoder(4, 1)
        vae_encoder = FlattenInputWrapper(vae_encoder)
        encoder = nn.Sequential(conv_encoder, vae_encoder)

        z, mu, sigma = encoder(x)

        conv_decoder = nn.ConvTranspose2d(1, 1, 2, 2)
        vae_decoder = LinearGaussianVAE.Decoder(4, 1)
        vae_decoder = ReshapeOutputWrapper(vae_decoder, (1, 2, 2))
        decoder = nn.Sequential(vae_decoder, conv_decoder)
        decoder = PassThroughWrapper(decoder)

        autoencoder = nn.Sequential(encoder, decoder)

        y, mu, sigma = autoencoder(x)

        assert y.shape == (10, 1, 4, 4)
        print(mu)
        print(sigma)



