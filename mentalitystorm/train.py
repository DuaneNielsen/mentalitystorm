import torch
import torch.utils.data as du
from mentalitystorm import Observable, TensorBoardObservable, config
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


BeforeArgs = namedtuple('BeforeArgs', 'trainer, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run')
AfterArgs = namedtuple('AfterArgs', 'trainer, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run,output, loss')


class SimpleTrainer:
    def __init__(self):
        self.context = {}
        self.before_hooks = []
        self.after_hooks = []

    def register_before_hook(self, func):
        """ Adds a closure to be executed before minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run'
        :return: nothing
        """
        self.before_hooks.append(func)

    def register_after_hook(self, func):
        """ Adds a closure to be executed after minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run, output, loss'
        :return: nothing
        """
        self.after_hooks.append(func)

    def execute_before(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run):
        before_args = BeforeArgs(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run)
        for closure in self.before_hooks:
            closure(before_args)

        # self.context['start'] = time.time()

    def execute_after(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run, output, loss):
        after_args = AfterArgs(self, payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run, output, loss)
        for closure in self.after_hooks:
            closure(after_args)

        # stop = time.time()
        # loop_time = stop - self.context['start']
        # self.writePerformanceToTB(loop_time, input_data.shape[0])

    def train(self, model, optimizer, lossfunc, dataloader, selector, run):
        device = config.device()
        model.to(device)
        model.train()

        for payload in dataloader:

            input_data = selector.get_input(payload, device)
            target_data = selector.get_target(payload, device)

            self.execute_before(payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run)

            optimizer.zero_grad()
            output = model(*input_data)
            if type(output) == tuple:
                loss = lossfunc(*output, *target_data)
            else:
                loss = lossfunc(output, *target_data)
            loss.backward()
            optimizer.step()

            self.execute_after(payload, input_data, target_data, model, optimizer, lossfunc, dataloader, selector, run, output, loss)

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
