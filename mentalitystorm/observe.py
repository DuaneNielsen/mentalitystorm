from abc import ABC, abstractmethod
from .image import NumpyRGBWrapper
from .config import config
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np


""" Dispatcher allows dipatch to views.
View's register here
To send a message, inherit Observable and use updateObservers
"""


class Dispatcher:
    def __init__(self):
        self.pipelineView = {}

    @staticmethod
    def registerView(tag, observer):
        if tag not in dispatcher.pipelineView:
            dispatcher.pipelineView[tag] = []
        dispatcher.pipelineView[tag].append(observer)

        return tag, len(dispatcher.pipelineView[tag]) -1

    @staticmethod
    def unregisterView(id):
        del dispatcher.pipelineView[id[0]][id[1]]

dispatcher = Dispatcher()

""" Observable provides dispatch method.
To use, make sure the object has a Dispatcher 
"""


class Observable:

    def updateObserversWithImage(self, tag, image, format=None, training=True, always_write=False):
        metadata = {}
        metadata['always'] = always_write
        metadata['func'] = 'image'
        metadata['name'] = tag
        metadata['format'] = format
        metadata['training'] = training
        self.updateObservers(tag, image, metadata)



    def updateObservers(self, tag, data, metadata=None):
        if tag not in dispatcher.pipelineView:
            dispatcher.pipelineView[tag] = []
        for observer in dispatcher.pipelineView[tag]:
            observer.update(data, metadata)

    @staticmethod
    def updateObserversWithImageStatic(self, tag, image, format=None, training=True, always_write=False):
        metadata = {}
        metadata['always'] = always_write
        metadata['func'] = 'image'
        metadata['name'] = tag
        metadata['format'] = format
        metadata['training'] = training
        self.updateObservers(tag, image, metadata)


    @staticmethod
    def updateObserversStatic(tag, data, metadata=None):
        if tag not in dispatcher.pipelineView:
            dispatcher.pipelineView[tag] = []
        for observer in dispatcher.pipelineView[tag]:
            observer.update(data, metadata)


    """ Sends a close event to all observers.
    used to close video files or save at the end of rollouts
    """
    def endObserverSession(self):
        for tag in dispatcher.pipelineView:
            for observer in dispatcher.pipelineView[tag]:
                observer.end_session()

    """ This is freaking brilliant, need more of this kind of thing!
    Attach to model.register_forward_hook(Observerable.send_output_as_image)
    """
    @staticmethod
    def send_output_as_image(self, input, output):
        obs = Observable()
        obs.updateObserversWithImage('output', output.data, 'tensorPIL')


class ImageChannel(Observable):
    def __init__(self, channel_id):
        self.channel_id = channel_id

    """ This is freaking brilliant, need more of this kind of thing!
    Attach to model.register_forward_hook(Observerable.send_output_as_image)
    """
    def send_output_as_image(self, object, input, output):
        self.updateObserversWithImage(self.channel_id, output.data, 'tensorPIL')


""" Abstract base class for implementing View.
"""


class View(ABC):

    @abstractmethod
    def update(self, data, metadata):
        raise NotImplementedError

    def endSession(self):
        pass


class ImageFileWriter(View):
    def __init__(self, directory, prefix, num_images=8192):
        super(ImageFileWriter, self).__init__()
        self.writer = None
        self.directory = directory
        self.prefix = prefix
        self.num_images = num_images
        self.imagenumber = 0

    def update(self, screen, metadata=None):

        in_format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        Image.fromarray(frame).save(self.directory + '/' + self.prefix + str(self.imagenumber) + '.png')
        self.imagenumber = (self.imagenumber + 1) % self.num_images



class OpenCV(View):
    def __init__(self, title='title', screen_resolution=(640,480)):
        super(OpenCV, self).__init__()
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution

    def update(self, screen, metadata=None):

        format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, format)
        frame = frame.getImage()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)


class ImageViewer:
    def __init__(self, title='title', screen_resolution=(640,480), format=None, channels=None):
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution
        self.format = format
        self.channels = channels

    def get_channels(self, input):
        if type(input) == torch.Tensor:
            channels = self.channels if self.channels is not None else range(input.shape[0])
            if len(channels) > 3:
                raise Exception("Too many channels, select the channels manually")
            frame = input[channels].data
            if frame.shape[0] == 2:
                # if 2 channels, put an extra channel in
                shape = 1, frame.shape[1], frame.shape[2]
                dummy_channel = torch.zeros(shape)
                frame = torch.cat((frame, dummy_channel), dim=0)
            return frame
        else:
            return input

    def view_input(self, model, input, output):
        image = input[0] if isinstance(input, tuple) else input
        self.update(image)

    def view_output(self, model, input, output):
        image = output[0] if isinstance(output, tuple) else output
        self.update(image)

    def strip_batch(self, image):
        if type(image) == torch.Tensor:
            if len(image.shape) == 4:
                return image[0]
            else:
                return image
        elif type(image) == np.ndarray:
            return image

    def update(self, screen):
        if type(screen) == torch.Tensor:
            screen = screen.cpu()
        frame = self.strip_batch(screen)
        frame = self.get_channels(frame)
        frame = NumpyRGBWrapper(frame, self.format)
        frame = frame.getImage()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)



class Plotter():

    def __init__(self, figure):
        self.image = None
        self.figure = figure
        plt.ion()

    def setInput(self, input):
        if input == 'numpyRGB':
            self.C = lambda x : x

    def update(self, screen, metadata=None):

        plt.figure(self.figure)

        image = self.C(screen)

        if self.image is None:
            self.image = plt.imshow(image)
        else:
            self.image.set_array(image)

        plt.pause(0.001)
        #plt.draw()


class TensorBoard(View, SummaryWriter):
    def __init__(self, run=None, comment='default', image_freq=50):
        View.__init__(self)
        SummaryWriter.__init__(self, run, comment)
        self.image_freq = image_freq
        self.dispatch = {'tb_step': self.step,
                         'tb_scalar': self.scalar,
                         'image': self.image,
                         'histogram': self.histogram,
                         'text': self.text
                         }
        self.global_step = None

    def register(self):
        Dispatcher.registerView('tb_step', self)
        Dispatcher.registerView('tb_training_loss', self)
        Dispatcher.registerView('tb_test_loss', self)
        Dispatcher.registerView('input', self)
        Dispatcher.registerView('output', self)
        Dispatcher.registerView('z', self)
        Dispatcher.registerView('tb_train_time', self)
        Dispatcher.registerView('tb_train_time_per_item', self)
        Dispatcher.registerView('BCELoss', self)
        Dispatcher.registerView('KLDLoss', self)
        Dispatcher.registerView('MSELoss', self)
        Dispatcher.registerView('histogram', self)
        Dispatcher.registerView('text', self)
        Dispatcher.registerView('z_cor_scalar', self)
        Dispatcher.registerView('z_corr', self)


    def update(self, data, metadata):
        func = self.dispatch.get(metadata['func'])
        func(data, metadata)


    def step(self, data, metadata):
        self.global_step = metadata['tb_global_step']

    def scalar(self, value, metadata):
        if self.global_step:
            self.add_scalar(metadata['name'], value, self.global_step)

    def image(self, value, metadata):
        if 'always' in metadata and metadata['always']:
            self.add_image(metadata['name'], value, self.global_step)
        if self.global_step and self.global_step % self.image_freq == 0 and not metadata['training']:
            self.add_image(metadata['name'], value, self.global_step)

    def histogram(self, data, metadata):
        tag = metadata['tag']
        step = metadata['step']
        if 'bins' in metadata:
            bins = metadata['bins']
            self.add_histogram(tag, data, step, bins)
        else:
            self.add_histogram(tag, data, step)

    def text(self, data, metadata):
        tag = metadata['tag']
        step = metadata['step']
        self.add_text(tag, data, step)

""" Convenience methods for dispatch to tensorboard
requires that the object also inherit Observable
"""

#todo replace this with a wrapper, and dispatch from there instead
# noinspection PyUnresolvedReferences
class TensorBoardObservable:
    def __init__(self):
        ## this isn't a great solution, need to come up with something better
        self.global_step = 0
        if hasattr(self, 'metadata') and 'tb_global_step' in self.metadata:
            self.global_step = self.metadata['tb_global_step']

    def tb_global_step(self):
        self.global_step += 1
        self.updateObservers('tb_step', None, {'func': 'tb_step', 'tb_global_step': self.global_step})
        if hasattr(self, 'metadata') and 'tb_global_step' in self.metadata:
            self.metadata['tb_global_step'] = self.global_step

    def writeScalarToTB(self, tag, value, tb_name):
        self.updateObservers(tag, value,
                             {'func': 'tb_scalar',
                              'name': tb_name})

    def writeTrainingLossToTB(self, loss):
        self.writeScalarToTB('tb_training_loss', loss, 'loss/train')

    def writeTestLossToTB(self, loss):
        self.writeScalarToTB('tb_test_loss', loss, 'loss/test')

    def writePerformanceToTB(self, time, batch_size):
        self.writeScalarToTB('tb_train_time', time, 'perf/train_time_per_batch')
        if batch_size != 0:
            self.writeScalarToTB('tb_train_time_per_item', time/batch_size, 'perf/train_time_per_item')

    def writeHistogram(self, label, data, epoch, bins=None):
        metadata = {}
        metadata['func'] = 'histogram'
        metadata['tag'] = label
        metadata['step'] = epoch
        if bins is not None:
            metadata['bins'] = bins
        self.updateObservers('histogram', data, metadata)

    def writeText(self, label, data, step):
        metadata = {}
        metadata['func'] = 'text'
        metadata['tag'] = label
        metadata['step'] = step
        self.updateObservers('text', data, metadata)


class SummaryWriterWithGlobal(SummaryWriter):
    def __init__(self, comment):
        super(SummaryWriterWithGlobal, self).__init__(comment=comment)
        self.global_step = 0

    def tensorboard_step(self):
        self.global_step += 1

    def tensorboard_scaler(self, name, scalar):
        self.add_scalar(name, scalar, self.global_step)

    """
    Adds a matplotlib plot to tensorboard
    """
    def plotImage(self, plot):
        self.add_image('Image', plot.getPlotAsTensor(), self.global_step)
        plot.close()
