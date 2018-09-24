from unittest import TestCase
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from mentalitystorm import Run, RunFac, DataPackage, SimpleTrainer, TestMSELoss
from tqdm import tqdm


class Selector:
    def get_input(self, payload, device):
        return (payload[0].to(device), )

    def get_target(self, payload, device):
        return ( payload[0].to(device), )


def print_loss_term(self, key, value):
    print('%s %f' % (key, value.item()))


def tb_loss_term(self, loss_term, value):
    self.run.tb.add_scalar('loss/' + loss_term, value.item(), self.run.step)


def print_loss(args):
    print('Loss %f' % args.loss.item())


def tb_loss(args):
    args.self.run.tb.add_scalar('loss/Train Loss', args.loss.item(), args.run.step)


class TestRunFac(TestCase):

    def test_initialzier(self):

        class FunkyClass:
            def __init__(self, groove, funkmode='maximum'):
                self.groove = groove
                self.funkmode = funkmode

            def funk(self):
                print(self.groove + ' is ' + self.funkmode)

        class GroovyClass:
            def __init__(self, groove):
                self.groove = groove

            def funk(self):
                print('Get doing the ' + self.groove)


        from mentalitystorm.runners import Init

        init_funky = Init(FunkyClass, 'funky_chiken', 'maximoso')
        funky = init_funky.construct()
        funky.funk()

        init_groovy = Init(GroovyClass, 'groovy_gibbon')
        funky = init_groovy.construct()
        funky.funk()

    def test_run(self):
        from mentalitystorm.runners import Init, Run
        from torch.optim import Adam

        model_i = Init(nn.Linear, 10, 10)
        opt_i = Init(Adam, lr=1e-3)
        loss_i = Init(TestMSELoss)

        x = torch.rand(100, 10)
        y = torch.rand(100, 10)
        dataset = data_utils.TensorDataset(x, y)
        data_package = DataPackage(dataset, Selector())

        run = Run(model_i, opt_i, loss_i, data_package)
        model, opt, loss_fn, data_package, trainer, tester, run = run.construct()

        loss_fn.register_hook(print_loss_term)
        loss_fn.register_hook(tb_loss_term)

        dev, train, test, selector = data_package.loaders(batch_size=2)
        for epoch in tqdm(run.for_epochs(10)):
            epoch.execute_before(epoch)
            trainer.train(model, opt, loss_fn, dev, selector, run, epoch)
            epoch.execute_after(epoch)


    def test_simplerunfac(self):
        from mentalitystorm.runners import Init, Run, SimpleRunFac
        from torch.optim import Adam

        x = torch.rand(100, 10)
        y = torch.rand(100, 10)
        dataset = data_utils.TensorDataset(x, y)
        data_package = DataPackage(dataset, Selector())

        runs = SimpleRunFac()
        runs.run_list.append(Run(Init(nn.Linear, 10, 10), Init(Adam, lr=1e-3), Init(TestMSELoss), data_package))

        for model, opt, loss_fn, data_package, trainer, tester, run in runs:

            dev, train, test, selector = data_package.loaders(batch_size=5)
            for epoch in tqdm(run.for_epochs(10)):
                epoch.execute_before(epoch)

                train_loss_hook = loss_fn.register_hook(print_loss_term)
                train_tb_loss_hook = loss_fn.register_hook(tb_loss_term)

                trainer.train(model, opt, loss_fn, dev, selector, run, epoch)

                train_loss_hook.remove()
                train_tb_loss_hook.remove()

                tester.test(model, loss_fn, dev, selector, run, epoch)

                epoch.execute_after(epoch)
