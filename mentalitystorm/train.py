import torch
from mentalitystorm.config import config
from mentalitystorm.util import Hookable
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


class SimpleTester(Hookable):

    def test(self, model, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.eval()
        model.epoch = epoch

        for payload in dataloader:

            input_data = selector.get_input(payload, device)
            target_data = selector.get_target(payload, device)

            before_args = BeforeArgs(self, payload, input_data, target_data, model, None, lossfunc, dataloader,
                                     selector, run, epoch)
            self.execute_before(before_args)

            output_data = model(*input_data)
            if type(output_data) == tuple:
                loss = lossfunc(*output_data, *target_data)
            else:
                loss = lossfunc(output_data, *target_data)

            after_args = AfterArgs(self, payload, input_data, target_data, model, None, lossfunc, dataloader,
                                   selector, run, epoch, output_data, loss)
            self.execute_after(after_args)

            run.step += 1


class SimpleInference(Hookable):
    def infer(self, model, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.eval()
        model.epoch = epoch

        for payload in dataloader:

            input_data = selector.get_input(payload, device)

            before_args = BeforeArgs(self, payload, input_data, None, model, None, lossfunc, dataloader,
                                     selector, run, epoch)
            self.execute_before(before_args)

            output_data = model(*input_data)

            after_args = AfterArgs(self, payload, input_data, None, model, None, lossfunc, dataloader,
                                   selector, run, epoch, output_data, None)
            self.execute_after(after_args)

            run.step += 1