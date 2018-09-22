from unittest import TestCase
import torch
import torch.nn as nn
import torch.utils.data as datautils
from mentalitystorm import config, Run, RunFac, DataPackage, SimpleTrainer
from tensorboardX import SummaryWriter


class Selector:
    def get_input(self, payload, device):
        return (payload[0].to(device), )

    def get_target(self, payload, device):
        return ( payload[0].to(device), )


class TestRunFac(TestCase):

    def test_runfac(self):


        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.rand(100, 10)
        y = torch.rand(100, 10)
        dataset = datautils.TensorDataset(x, y)
        data_package = DataPackage(dataset, Selector())
        loss_func = nn.MSELoss()
        dfr = Run(model=model, opt=optimizer, loss_fn=loss_func, data_package=data_package)

        rf = RunFac(dfr)

        def print_loss(args):
            print('Loss %f' % args.loss.item())

        def tb_loss(args):
            args.run.metadata['tb'].add_scalar('Train Loss', args.loss.item(), args.run.step)

        trainer = SimpleTrainer()
        trainer.register_after_hook(print_loss)
        trainer.register_after_hook(tb_loss)

        for model, opt, loss_fn, data_package, run in rf:
            run.metadata['tb'] = SummaryWriter(config.tb_run_dir(model))

            dev, train, test, selector = data_package.loaders(batch_size=2)
            for epoch in range(100):
                trainer.train(model, opt, loss_fn, dev, selector, run)
                # for x, y in train:
                #     optimizer.zero_grad()
                #     y_pred = model(x)
                #     loss = loss_fn(x, y_pred)
                #     print('loss %f' % loss.item())
                #     loss.backward()
                #     optimizer.step()
