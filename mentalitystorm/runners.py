from abc import ABC

import torch
from tqdm import tqdm

from mentalitystorm import Storeable, MSELoss, config, TensorBoard, OpenCV, ElasticSearchUpdater, \
    Dispatcher, Observable, TensorBoardObservable

import numpy as np

class Trainer(ABC, Observable, TensorBoardObservable):
    def __run__(self, model, dataset, batch_size, lossfunc, optimizer, epochs=2):
        device = config.device()
        if isinstance(model, Storeable):
            run_name = config.run_id_string(model)
            model.metadata['run_name'] = run_name
            model.metadata['run_url'] = config.run_url_link(model)
            model.metadata['git_commit_hash'] = config.GIT_COMMIT
            model.metadata['dataset'] = str(dataset.root)
            model.metadata['loss_class'] = type(lossfunc).__name__

        for epoch in tqdm(range(epochs)):
            model.train_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc, optimizer=optimizer)
            losses, histograms = model.test_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc)

            if losses is None:
                raise Exception('Test loop did not run, this is probably because there is not a full batch,'
                                'decrease batch size and try again')

            l = torch.Tensor(losses)

            ave_test_loss = l.mean().item()
            import math
            if not math.isnan(ave_test_loss):
                model.metadata['ave_test_loss'] = ave_test_loss

            for i, histogram in enumerate(np.rollaxis(histograms, 1)):
                self.writeHistogram('latentvar' + str(i), histogram, epoch)

            if 'epoch' not in model.metadata:
                model.metadata['epoch'] = 1
            else:
                model.metadata['epoch'] += 1
            model.save()

    def __demo__(self, model, dataset):
        device = config.device()
        model.demo_model(dataset, 1, device)


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