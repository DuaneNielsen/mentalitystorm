import torch
from mentalitystorm.config import config
from mentalitystorm.util import Hookable
from collections import namedtuple





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