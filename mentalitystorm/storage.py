import pickle
from pathlib import Path
import inspect
import hashlib
import logging
from mentalitystorm.config import config
from .config import slug

log = logging.getLogger('Storage')

"""Stores the object params for initialization
Storable MUST be the first in the inheritance chain
So put it as the first class in the inheritance
ie: class MyModel(Storable, nn.Module)
the init method must also be called as the LAST one in the sequence..
ie: nn.Module.__init__(self)
    Storable.__init(self, arg1, arg2, etc)
fixing to make less fragile is on todo, but not trivial...
"""
class Storeable():
    def __init__(self):
        self.classname = type(self)

        #snag the args from the child class during initialization
        stack = inspect.stack()
        child_callable = stack[1][0]
        argname, _, _, argvalues = inspect.getargvalues(child_callable)

        self.repr_string = ""
        arglist = []
        for key in argname:
            if key != 'self':
                self.repr_string += ' (' + key + '): ' + str(argvalues[key])
                arglist.append(argvalues[key])

        self.args = tuple(arglist)
        self.metadata = {}
        self.metadata['guid'] = self.guid()
        self.metadata['class_guid'] = self.class_guid()
        self.metadata['classname'] = type(self).__name__
        self.metadata['args'] = self.repr_string
        self.metadata['repr'] = repr(self)
        self.metadata['slug'] = slug(self)


    def extra_repr(self):
        return self.repr_string


    """computes a unique GUID for each model/args instance
    """
    def guid(self):
        import random, string
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

    """computes a unique GUID for each model/args pair
    """
    def class_guid(self):
        md5 = hashlib.md5()
        md5.update(self.repr_string.encode('utf8'))
        return md5.digest().hex()


    """ makes it so we only save the init params and weights to disk
    the res
    """
    def __getstate__(self):
        save_state = []
        save_state.append(self.metadata)
        save_state.append(self.args)
        save_state.append(self.state_dict())
        return save_state

    """ initializes a fresh model from disk with weights
    """
    def __setstate__(self, state):
        log.debug(state)
        self.__init__(*state[1])
        self.metadata = state[0]
        self.load_state_dict(state[2])

    def save(self, filename=None):
        path = Path(filename)
        self.metadata['filename'] = path.name
        from datetime import datetime
        self.metadata['timestamp'] = datetime.utcnow()
        self.metadata['parameters'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            metadata, args, state_dict = self.__getstate__()
            pickle.dump(metadata, f)
            pickle.dump(self, f)
        return path.name

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            try:
                _ = pickle.load(f)
                model = pickle.load(f)
            except Exception as e:
                message = "got exception when loading {}".format(filename)
                log.error(message)
                log.error(e)
                raise
            return model


    """ Load metadata only
    """
    @staticmethod
    def load_metadata(filename, data_dir=None):
        with Storeable.fn(filename, data_dir).open('rb') as f:
            return  pickle.load(f)

    @staticmethod
    def update_metadata(filename, metadata_dict, data_dir=None):
        """ Load model from disk and flag it as reloaded """
        assert type(metadata_dict) is dict
        model = Storeable.load(filename, data_dir)
        model.metadata = metadata_dict
        model.save(filename, data_dir)


class ModelDb:
    def __init__(self):
        self.metadatas = []
        self.datapath = config.modelpath()
        for file in self.datapath.iterdir():
            self.metadatas.append(Storeable.load_metadata(file.name))

    def print_data(self):
        for metadata in self.metadatas:
            for field, value in metadata.items():
                print(field, value)

    def print_data_for(self, filename):
        metadata = Storeable.load_metadata(filename)
        for field, value in metadata.items():
            print(field, value)


    """ Returns the 2 best results for each guid
    """
    def topNLossbyModelGuid(self, n):
        import collections
        Loss = collections.namedtuple('Loss', 'loss metadata')
        model_top = {}
        for model in self.metadatas:
            guid =  model['guid']
            ave_test_loss = model['ave_test_loss'] if 'ave_test_loss' in model else None
            if ave_test_loss is not None:
                if guid not in model_top:
                    model_top[guid] = []
                model_top[guid].append((ave_test_loss, model))

        for guid in model_top:
            model_top[guid].sort(key=lambda tup: tup[0])
            model_top[guid] = model_top[guid][0:n]

        return model_top


    """ returns the model with the best loss
    """
    def best_loss_for_model_class(self, modelclassname):
        import sys
        best_loss = sys.float_info.max
        checkpoint_file = None
        for metad in self.metadatas:
            if metad['classname'] == modelclassname:
                if metad['ave_test_loss'] < best_loss:
                    checkpoint_file = metad['filename']

        return checkpoint_file



    """ syncs data in filesystem to elastic
    Dumb sync, just drops the whole index and rewrites it
    """
    def sync_to_elastic(self, host='localhost', port=9200):
        from elasticsearch import ElasticsearchException
        from mentalitystorm.elastic import ElasticSetup

        es = ElasticSetup(host, port)
        es.deleteModelIndex()
        es.createModelIndex()

        for metadata in self.metadatas:
            try:
                res = es.es.index(index="models", doc_type='model', id=metadata['filename'], body=metadata)
                print(res)
            except ElasticsearchException as es1:
                print(es1)











