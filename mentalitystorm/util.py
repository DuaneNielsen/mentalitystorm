import torch
import collections
import weakref


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = floor( ((h_w[0] + (2 * pad[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_padding=0):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1) * stride - (2 * pad) + kernel_size[0] + output_padding
    w = (h_w[1] - 1) * stride - (2 * pad) + kernel_size[1] + output_padding
    return h, w


def default_maxunpool_indices(output_shape, kernel_size, batch_size, channels, device):
    """ Generates a default index map for nn.MaxUnpool2D operation.
    output_shape: the shape that was put into the nn.MaxPool2D operation
    in terms of nn.MaxUnpool2D this will be the output_shape
    pool_size: the kernel size of the MaxPool2D
    """

    ph = kernel_size[0]
    pw = kernel_size[1]
    h = output_shape[0]
    w = output_shape[1]
    ih = output_shape[0] // 2
    iw = output_shape[1] // 2
    h_v = torch.arange(ih, dtype=torch.int64, device=device) * pw * ph * iw
    w_v = torch.arange(iw, dtype=torch.int64, device=device) * pw
    h_v = torch.transpose(h_v.unsqueeze(0), 1,0)
    return (h_v + w_v).expand(batch_size, channels, -1, -1)


class Handles:
    def __init__(self):
        self.handles = []

    def __iadd__(self, removable_handle):
        self.handles.append(removable_handle)
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return (self.hooks_dict_ref(), self.id)

    def __setstate__(self, state):
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(collections.OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()


class Hookable:
    """
    Adds entry points for removable hooks
    """
    def __init__(self):
        self.context = {}
        self.before_hooks = collections.OrderedDict()
        self.after_hooks = collections.OrderedDict()

    def register_before_hook(self, func):
        """ Adds a closure to be executed before minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run'
        :return: a handle to remove the hook
        """

        handle = RemovableHandle(self.before_hooks)
        self.before_hooks[handle.id] = func
        return handle

    def register_after_hook(self, func):
        """ Adds a closure to be executed after minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run, output, loss'
        :return: a handle to remove the hook
        """
        handle = RemovableHandle(self.after_hooks)
        self.after_hooks[handle.id] = func
        return handle

    def execute_before(self, before_args):

        for closure in self.before_hooks.values():
            closure(before_args)

        # self.context['start'] = time.time()

    def execute_after(self, after_args):
        for closure in self.after_hooks.values():
            closure(after_args)

        # stop = time.time()
        # loop_time = stop - self.context['start']
        # self.writePerformanceToTB(loop_time, input_data.shape[0])