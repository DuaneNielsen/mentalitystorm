from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from mentalitystorm import Storeable, MSELoss, config, TensorBoard, OpenCV, ElasticSearchUpdater, \
    Dispatcher, Observable, TensorBoardObservable
import numpy as np
from tabulate import tabulate
from collections import namedtuple
import torch.utils.data as data_utils
from tensorboardX import SummaryWriter
from .util import Hookable
from .losses import Lossable
from .train import SimpleTrainer, SimpleTester
from pathlib import Path


class Selector(ABC):
    @abstractmethod
    def get_input(self, package, device):
        raise NotImplementedError

    @abstractmethod
    def get_target(self, package, device):
        raise NotImplementedError


DataSets = namedtuple('DataSets', 'dev, train, test')
DataLoaders = namedtuple('DataSets', 'dev, train, test')


class Splitter:

    def split(self, dataset):
        """
        Splits the dataset into dev, train and test
        :param dataset: the dataset to split
        :return: DataSets named tupple (dev, train, test)
        """
        dev = data_utils.Subset(dataset, range(len(dataset) * 2 // 10))
        train = data_utils.Subset(dataset, range(0, len(dataset) * 9//10))
        test = data_utils.Subset(dataset, range(len(dataset) * 9 // 10 + 1, len(dataset)))
        return DataSets(dev=dev, train=train, test=test)

    def loaders(self, dataset, **kwargs):
        """ Returns a named tuple of loaders

        :param kwargs: same as kwargs for torch.utils.data.Loader
        :return: named tuple of dataloaders, dev, train, test
        """

        split = self.split(dataset)

        dev = data_utils.DataLoader(split.dev, **kwargs)
        train = data_utils.DataLoader(split.train, **kwargs)
        test = data_utils.DataLoader(split.test, **kwargs)

        return DataLoaders(dev=dev, train=train, test=test)


class DataPackage:
    """ Datapackage provides everything required to load data to train and test a model
    """
    def __init__(self, dataset, selector, splitter=None):
        """
        :param dataset: the torch dataset object
        :param selector: selects the model input and model target data
        :param splitter: splits data into dev, train and test sets
        """
        self.dataset = dataset
        self.splitter = splitter if splitter is not None else Splitter()
        self.selector = selector

    def loaders(self, **kwargs):
        dev, train, test = self.splitter.loaders(self.dataset, **kwargs)
        return dev, train, test, self.selector


class Epoch(Hookable):
    def __init__(self, index, run):
        super(Epoch, self).__init__()
        self.ix = index
        self.run = run


class EpochIter:
    def __init__(self, num_epochs, run):
        self.num_epochs = num_epochs
        self.run = run
        self.last_epoch = run.epochs + num_epochs

    def __iter__(self):
        return self

    def __next__(self):
        if self.run.epochs == self.last_epoch:
            raise StopIteration
        epoch = Epoch(self.run.epochs, self.run)
        self.run.epoch = epoch
        self.run.epochs += 1
        return epoch

        #torch.nn.init.kaiming_uniform_(m.weight)
        #m.bias.data.fill_(0.01)


class Init:
    def construct(self, model=None):
        return NotImplementedError


class Params(Init):
    def __init__(self, clazz, *args, **kwargs):
        self.clazz = clazz
        self.args = args
        self.kwargs = kwargs

    def construct(self, model=None):
        if model is None:
            return self.clazz(*self.args, **self.kwargs)
        else:
            return self.clazz(model.parameters(), *self.args, **self.kwargs)


class LoadModel(Init):
    def __init__(self, filename):
        self.filename = filename

    def construct(self, model=None):
        return Storeable.load(self.filename)


class Run:
    def __init__(self, model, opt, loss_fn, data_package, trainer=None, tester=None, run_name=None, tensorboard=True,
                 weights_init_func=None):
        """
        :param model: the model to train
        :param opt: the optimizer for the model, should already be initialized with model params
        :param loss_fn: the loss function to use for training
        :param data_package: specifies what to load, how to split the dataset, and what is inputs and targets
        :param trainer: the trainer to use
        :param tester: the tester to use
        :param run_name name of the run
        :param tensorboard: to use tensorboard or not
        :param weights_init_func: to initialize the model weights
        """

        if not isinstance(model, Init):
            raise Exception('model should be a Initializer, dont put a naked model in!')

        if opt is not None and not isinstance(opt, Init):
            raise Exception('model should be a Initializer, dont put a naked optimizer in!')

        self.model_i = model
        self.model = None
        self.model_params = None

        self.opt_i = opt
        self.opt = None

        self.loss_fn_i = loss_fn
        self.loss = None

        self.data_package = data_package

        self.trainer = trainer if trainer is not None else SimpleTrainer()
        self.trainer.run = self

        self.tester = tester if tester is not None else SimpleTester()
        self.tester.run = self

        self.run_name = run_name
        self.run_id = None
        self.epochs = 0
        self.step = 0
        self.epoch = None

        self.context = {}

        self.weights_init_func = weights_init_func

    def construct_model_and_optimizer(self):
        self.model = self.model_i.construct()
        self.inject_modules(self.model)
        if self.model_params is not None:
            self.model.load_state_dict(self.model_params)
        elif self.weights_init_func is not None:
            self.model.apply(self.weights_init_func)
        if self.opt is not None:
            self.opt = self.opt_i.construct(self.model)
            self.opt.run = self
        return self.model, self.opt

    def construct_loss(self):
        if isinstance(self.loss_fn_i, Params):
            self.loss = self.loss_fn_i.construct()
        else:
            self.loss = self.loss_fn_i
        if isinstance(self.loss, Lossable):
            self.loss.run = self
        return self.loss

    def inject_modules(self, model):
        if model is not None:
            for model in model.modules():
                model.run = self

    def init_run_dir(self, model, increment_run=True, tensorboard=True):
        if increment_run:
            config.increment_run_id()
        if self.run_name is None:
            self.run_id = 'runs/' + config.rolling_run_number() + '/' + config.slug(model)
        else:
            self.run_id = 'runs/' + config.rolling_run_number() + '/' + self.run_name
        self.tb = SummaryWriter(config.tb_run_dir(self.run_id)) if tensorboard else None

    def construct(self, increment_run=True, tensorboard=True, data_package=None):
        self.data_package = data_package if self.data_package is None else self.data_package
        self.construct_model_and_optimizer()
        self.construct_loss()
        self.init_run_dir(self.model, increment_run, tensorboard)
        return self.model, self.opt, self.loss, self.data_package, self.trainer, self.tester, self

    def for_epochs(self, num_epochs):
        return EpochIter(num_epochs, self)

    def __getstate__(self):
        state = self.__dict__.copy()
        if state['model'] is not None:
            state['model_params'] = self.model.state_dict()
        state['model'] = None
        state['opt'] = None
        state['loss'] = None
        state['data_package'] = None
        state['tb'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self):
        file = config.datapath(self.run_id + '/epoch' + '%04d' % self.epochs + '.run')
        import pickle
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        import pickle
        with open(file, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def resume(file, data_package, increment_run=False):
        run = Run.load(file)
        return run.construct(increment_run=increment_run, data_package=data_package)


class SimpleRunFac:
    def __init__(self):
        self.run_list = []
        self.data_package = None
        self.resuming = False

    def __iter__(self):
        self.run_id = 0
        if not self.resuming:
            config.increment_run_id()
        return self

    def __next__(self):
        if self.run_id == len(self.run_list):
            raise StopIteration
        run = self.run_list[self.run_id].construct(increment_run=False, data_package=self.data_package)
        self.run_id += 1
        return run

    @staticmethod
    def resume(run_dir, data_package, resuming=True):
        run_fac = SimpleRunFac()
        run_fac.resuming = resuming
        run_fac.data_package = data_package
        dir = Path(run_dir)
        for subdir in dir.glob('*'):
            if subdir.is_dir():
                # super pythonic (and completely unreadable) way to get the last epoch in the run
                last_epoch = sorted([f for f in subdir.glob('*.run')], reverse=True)[0]
                run_fac.run_list.append(Run.load(last_epoch.absolute()))
        return run_fac

    @staticmethod
    def reuse(run_dir, data_package):
        """
        Reuses an existing run on a new datasset
        :param run_dir:
        :param data_package:
        :return:
        """
        return SimpleRunFac.resume(run_dir, data_package, False)

ModelOpt = namedtuple('ModelOpt', 'model, opt')

RunParams = namedtuple('RunParams', 'model, opt, loss_fn, data_package, trainer, tester, run_name, tensorboard')


class RunFac:
    def __init__(self, model=None, opt=None, loss_fn=None, data_package=None, trainer=None, tester=None, run_name=None,
                 tensorboard=True):
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.data_package = data_package
        self.trainer = trainer
        self.tester = tester
        self.run_name = run_name
        self.tensorboard = tensorboard

        self._model_opts = []
        self.data_packages = []
        self.loss_fns = []
        self.run_id = 0
        self.run_list = []

    def add_model_opt(self, model, opt):
        self._model_opts.append(ModelOpt(model, opt))

    def all_empty(self):
        return len(self._model_opts) + len(self.data_packages) + len(self.loss_fns) == 0

    def build_run(self):
        if self.all_empty():
            self.run_list.append(RunParams(model=self.model,
                                           opt=self.opt,
                                           loss_fn=self.loss_fn,
                                           data_package=self.data_package,
                                           trainer=self.trainer,
                                           tester=self.tester,
                                           run_name=self.run_name,
                                           tensorboard=self.tensorboard))

        for loss_fn in self.loss_fns:
            if isinstance(loss_fn, tuple):
                self.run_list.append(RunParams(model=self.model,
                                               opt=self.opt,
                                               loss_fn=loss_fn[1],
                                               data_package=self.data_package,
                                               trainer=self.trainer,
                                               tester=self.tester,
                                               run_name=loss_fn[0],
                                               tensorboard=self.tensorboard))
            else:
                self.run_list.append(RunParams(model=self.model,
                                               opt=self.opt,
                                               loss_fn=loss_fn,
                                               data_package=self.data_package,
                                               trainer=self.trainer,
                                               tester=self.tester,
                                               run_name=None,
                                               tensorboard=self.tensorboard))

    def __iter__(self):
        self.run_id = 0
        self.build_run()
        return self

    def __next__(self):
        if self.run_id == len(self.run_list):
            raise StopIteration
        run = Run(*self.run_list[self.run_id])
        self.run_id += 1
        return run.model, run.opt, run.loss_fn, run.data_package, run.trainer, run.tester, run

    def __getitem__(self, item):
        self.build_run()
        return self.run_list[item]


class Trainer(ABC, Observable, TensorBoardObservable):
    def __run__(self, model, dataset, batch_size, lossfunc, optimizer, epochs=2):
        device = config.device()
        self.init_run_data(dataset, lossfunc, model)

        for epoch in tqdm(range(epochs)):
            model.train_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc, optimizer=optimizer)
            losses, histograms = model.test_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc)

            self.update_ave_test_loss(losses, model)

            self.send_histogram(epoch, histograms)

            self.increment_epoch(model)
            model.save(config.model_fn(model))

    def __test__(self, model, dataset, batch_size, lossfunc, epochs):
        device = config.device()
        self.init_run_data(dataset, lossfunc, model)

        for epoch in tqdm(range(epochs)):

            losses, histograms = model.test_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc)
            self.update_ave_test_loss(losses, model)
            self.send_histogram(epoch, histograms)
            self.increment_epoch(model)

    def send_histogram(self, epoch, histograms):
        histograms = np.rollaxis(histograms, 1)
        for i, histogram in enumerate(histograms):
            self.writeHistogram('latentvar' + str(i), histogram, epoch)

        cor = np.corrcoef(histograms)

        table = ''
        for i, histogram in enumerate(histograms):
            table = table + '|' + str(i)
        table = table + '|\n'
        table = table + tabulate(cor, tablefmt='pipe')
        self.writeText('z_corr', table, epoch)

        # take the lower triangle
        I = np.identity(cor.shape[0])
        cor_scalar = cor - I
        cor_scalar = np.square(cor)
        cor_scalar = np.sqrt(cor_scalar)
        cor_scalar = np.sum(cor_scalar) /cor_scalar.size / 2.0
        self.writeScalarToTB('z_cor_scalar', cor_scalar, 'z/z_ave_correlation')

        image = np.expand_dims(cor, axis=0)
        image = np.square(image)
        image = np.sqrt(image)
        self.updateObserversWithImage('z_corr', image, always_write=True)

    def init_run_data(self, dataset, lossfunc, model):
        if isinstance(model, Storeable):
            run_name = config.run_id_string(model)
            model.metadata['run_name'] = run_name
            model.metadata['run_url'] = config.run_url_link(model)
            model.metadata['git_commit_hash'] = config.GIT_COMMIT
            model.metadata['dataset'] = str(dataset.root)
            model.metadata['loss_class'] = type(lossfunc).__name__
            #todo add loss parameters here

    def update_ave_test_loss(self, losses, model):
        if losses is None:
            raise Exception('Test loop did not run, this is probably because there is not a full batch,'
                            'decrease batch size and try again')
        l = torch.Tensor(losses)
        ave_test_loss = l.mean().item()
        import math
        if not math.isnan(ave_test_loss):
            model.metadata['ave_test_loss'] = ave_test_loss

    def increment_epoch(self, model):
        if 'epoch' not in model.metadata:
            model.metadata['epoch'] = 1
        else:
            model.metadata['epoch'] += 1

    def __demo__(self, model, dataset):
        device = config.device()
        model.demo_model(dataset, 1, device)

    def __sample__(self, model, z_dims):
        model.sample_model(z_dims)


#todo list test runner and matrix test runner
class ModelFactoryTrainer(Trainer):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model_args = []
        self.model_args_index = 0
        self.optimizer_type = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.model_args_index < len(self.model_args):
            model = self.model_type(*self.model_args[self.model_args_index])
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            self.model_args_index += 1
            return model, optim
        else:
            raise StopIteration()

    def run(self, dataset, batch_size, lossfunc=MSELoss, epochs=2):
        for model, optimizer in self:
            self.__run__(model, dataset, batch_size, lossfunc, optimizer, epochs)


class OneShotTrainer(Trainer):
    def run(self, model, dataset, batch_size, lossfunc, optimizer, epochs=2):
        self.__run__(model, dataset, batch_size, lossfunc, optimizer, epochs)


""" Runner with good defaults
Uses batch size 16, runs for 10 epochs, using MSELoss and Adam Optimizer with lr 0.001
"""


class OneShotEasyTrainer(Trainer):
    def run(self, model, dataset, batch_size=16, epochs=10, lossfunc=None):
        config.increment('run_id')

        tb = TensorBoard(config.tb_run_dir(model))
        tb.register()

        Dispatcher.registerView('input', OpenCV('input', (320, 420)))
        Dispatcher.registerView('output', OpenCV('output', (320, 420)))
        ElasticSearchUpdater().register()

        if lossfunc is None:
            lossfunc = MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.__run__(model, dataset, batch_size, lossfunc, optimizer, epochs)


class Demo(Trainer):
    def demo(self, model, dataset):
        Dispatcher.registerView('input', OpenCV('input', (320, 420)))
        Dispatcher.registerView('output', OpenCV('output', (320, 420)))
        self.__demo__(model, dataset)

    def sample(self, model, z_dims, samples=1):
        Dispatcher.registerView('sample_image', OpenCV('sample', (320, 420)))
        for _ in range(samples):
            self.__sample__(model, z_dims)
            import time
            time.sleep(1)

    def rotate(self, model, z_dims):
        Dispatcher.registerView('sample_image', OpenCV('sample', (320, 420)))
        for dim in range(z_dims):
            z = torch.full((z_dims,), 1.0).to(config.device())
            print('rotating ' + str(dim))
            for interval in np.linspace(-16.0, 16.0, 100):
                mag = abs(interval)
                sign = np.sign(interval)

                z[dim] = 1.0 + (sign * np.sqrt(mag))
                model.decode_model(z)
                import time
                time.sleep(0.05)

    def test(self, model, dataset, batch_size, lossfunc, epochs=2):
        Dispatcher.registerView('input', OpenCV('input', (320, 420)))
        Dispatcher.registerView('output', OpenCV('output', (320, 420)))

        self.__test__(model, dataset, batch_size, lossfunc, epochs)

