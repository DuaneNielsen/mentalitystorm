from unittest import TestCase
import torch
import torch.nn as nn
import torch.utils.data as datautils
from mentalitystorm import Run, RunFac, DataPackage, SimpleTrainer, TestMSELoss
from tqdm import tqdm


class Selector:
    def get_input(self, payload, device):
        return (payload[0].to(device), )

    def get_target(self, payload, device):
        return ( payload[0].to(device), )


class TestRunFac(TestCase):

    def test_runfac(self):

        def print_loss_term(self, key, value):
            print('%s %f' % (key, value.item()))

        def tb_loss_term(self, loss_term, value):
            self.run.tb.add_scalar('loss/' + loss_term, value.item(), run.step)

        def print_loss(args):
            print('Loss %f' % args.loss.item())

        def tb_loss(args):
            args.self.run.tb.add_scalar('loss/Train Loss', args.loss.item(), args.run.step)

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.rand(100, 10)
        y = torch.rand(100, 10)
        dataset = datautils.TensorDataset(x, y)
        data_package = DataPackage(dataset, Selector())

        loss_func = TestMSELoss()
        loss_func.register_term_hook(print_loss_term)
        loss_func.register_term_hook(tb_loss_term)

        trainer = SimpleTrainer()
        trainer.register_after_hook(print_loss)
        trainer.register_after_hook(tb_loss)

        dfr = Run(model=model, opt=optimizer, loss_fn=loss_func, data_package=data_package, trainer=trainer)

        rf = RunFac(dfr)

        for model, opt, loss_fn, data_package, trainer, tester, run in rf:
            dev, train, test, selector = data_package.loaders(batch_size=2)
            for epoch in tqdm(run.for_epochs(10)):
                trainer.train(model, opt, loss_fn, dev, selector, run, epoch)