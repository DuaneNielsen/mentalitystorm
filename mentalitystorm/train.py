import torch
import torch.utils.data as du
from mentalitystorm import Observable, TensorBoardObservable, config
from mentalitystorm.util import Hookable
from .losses import Lossable
import time
import numpy as np
from collections import namedtuple


class Checkable():
    def __init__(self):
        pass

    """ Builds a random variable for grad_check.
    shape: a tuple with the shape of the input
    batch=True will create a batch of 2 elements, useful if network has batchnorm layers    
    """
    @staticmethod
    def build_input(shape, batch=False):
        from torch.autograd import Variable
        input_shape = shape
        if batch:
            input_shape = (2, *shape)
        return Variable(torch.randn(input_shape).double(), requires_grad=True)

    """Runs a grad check.
    """
    def grad_check(self, *args):
        from torch.autograd import gradcheck
        gradcheck(self.double(), *args, eps=1e-6, atol=1e-4)


BeforeArgs = namedtuple('BeforeArgs', 'self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, '
                                      'selector, run, epoch')
AfterArgs = namedtuple('AfterArgs', 'self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, '
                                    'selector, run, epoch, output_data, loss')


class SimpleTrainer(Hookable):

    def train(self, model, optimizer, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.train()
        model.epoch = epoch

        for payload in dataloader:

            input_data = selector.get_input(payload, device)
            target_data = selector.get_target(payload, device)

            before_args = BeforeArgs(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader,
                                     selector, run, epoch)
            self.execute_before(before_args)

            optimizer.zero_grad()
            output_data = model(*input_data)
            if type(output_data) == tuple:
                loss = lossfunc(*output_data, *target_data)
            else:
                loss = lossfunc(output_data, *target_data)
            loss.backward()
            optimizer.step()

            after_args = AfterArgs(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader,
                                   selector, run, epoch, output_data, loss)
            self.execute_after(after_args)

            run.step += 1


class Trainable(Observable, TensorBoardObservable):

    @staticmethod
    def loader(dataset, batch_size):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def train_model(self, dataset, batch_size, device, lossfunc, optimizer):
        assert isinstance(lossfunc, Lossable)
        self.to(device)
        self.train()
        train_set = du.Subset(dataset, range(len(dataset) // 10, len(dataset) -1))
        train_loader = self.loader(train_set, batch_size)

        for batch_idx, (data, target) in enumerate(train_loader):
            start = time.time()
            data = data.to(device)
            optimizer.zero_grad()
            output = self(data)
            if type(output) == tuple:
                loss = lossfunc.loss(*output, data)
            else:
                loss = lossfunc.loss(output, data)
            self.writeTrainingLossToTB(loss/data.shape[0])
            loss.backward()
            optimizer.step()
            self.tb_global_step()
            stop = time.time()
            loop_time = stop - start
            self.writePerformanceToTB(loop_time, data.shape[0])

    def test_model(self, dataset, batch_size, device, lossfunc):
        assert isinstance(lossfunc, Lossable)
        with torch.no_grad():
            self.eval()
            self.to(device)
            test_set = du.Subset(dataset, range(0,len(dataset)//10))
            test_loader = self.loader(test_set, batch_size)
            losses = []

            hist = None

            for batch_idx, (data, target) in enumerate(test_loader):
                start = time.time()
                data = data.to(device)
                output = self(data)
                if type(output) == tuple:
                    loss = lossfunc.loss(*output, data)
                    z = output[1]
                    latentvar = z.squeeze().cpu().numpy()
                    if hist is None:
                        hist = latentvar
                    else:
                        hist = np.append(hist, latentvar, axis=0)
                else:
                    loss = lossfunc.loss(output, data)

                losses.append(loss.item())

                self.writeTestLossToTB(loss/data.shape[0])
                self.tb_global_step()
                stop = time.time()
                loop_time = stop - start
                self.writePerformanceToTB(loop_time, data.shape[0])

            return losses, hist

    def demo_model(self, dataset, batch_size, device):
        with torch.no_grad():
            self.eval()
            self.to(device)
            test_set = du.Subset(dataset, range(0, len(dataset) // 10))
            test_loader = self.loader(test_set, batch_size)

            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                output = self(data)
                import time
                time.sleep(1)

    def sample_model(self, z_dims):
        with torch.no_grad():
            self.eval()
            self.to(config.device())
            eps = torch.randn(z_dims).to(config.device())
            eps = eps.view(1, -1, 1, 1)
            image = self.sample(eps)

    def decode_model(self, z):
        with torch.no_grad():
            self.eval()
            self.to(config.device())
            eps = z.view(1, -1, 1, 1)
            return self.sample(eps)
