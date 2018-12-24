import torch


class Checker():
    def __init__(self, model):
        self.model = model

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
        gradcheck(self.model.double(), args, eps=1e-6, atol=1e-4)