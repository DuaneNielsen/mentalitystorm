from abc import ABC
import torch
from tqdm import tqdm
from mentalitystorm import Storeable, MSELoss, config, TensorBoard, OpenCV, ElasticSearchUpdater, \
    Dispatcher, Observable, TensorBoardObservable
import numpy as np
from tabulate import tabulate


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
            model.save()

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
