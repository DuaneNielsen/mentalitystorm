from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from .observe import Observable, TensorBoardObservable


class Lossable(ABC):
    @abstractmethod
    def loss(self, recon_x, mu, logvar, x): raise NotImplementedError


class BceKldLoss(Lossable, Observable, TensorBoardObservable):
    def __init__(self):
        Observable.__init__(self)
        TensorBoardObservable.__init__(self)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x)

        self.writeScalarToTB('BCELoss', BCE.item(), 'loss/BCELoss')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.writeScalarToTB('KLDLoss', KLD.item(), 'loss/KLDLoss')

        return BCE + KLD



class MseKldLoss(Lossable, Observable, TensorBoardObservable):
    def __init__(self, beta=1.0):
        self.beta = beta
        Observable.__init__(self)
        TensorBoardObservable.__init__(self)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        MSE = F.mse_loss(recon_x, x, reduction='sum')

        self.writeScalarToTB('MSELoss', MSE.item(), 'loss/MSELoss')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        # https: // openreview.net / forum?id = Sy2fzU9gl
        KLD = KLD * self.beta

        self.writeScalarToTB('KLDLoss', KLD.item(), 'loss/KLDLoss')

        return MSE + KLD

class BcelKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BceLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE #+ KLD


class MSELoss(Lossable):
    def loss(self, recon_x, mu, logvar, x):
        return F.mse_loss(recon_x, x)