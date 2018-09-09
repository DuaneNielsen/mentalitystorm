from unittest import TestCase
from models import AtariConv_v6
from mentalitystorm import Storeable
from mentalitystorm import ModelDb

class Restoreable(Storeable):
    def __init__(self, one, two):
        self.one = one
        self.two = two
        Storeable.__init__(self, one, two)

    def state_dict(self):
        return None

    def load_state_dict(self, thdict):
        pass

class TestStorable(TestCase):
    def test_params(self):
        filter_stack = [128, 128, 64, 64, 64]
        model = AtariConv_v6(filter_stack)

    def test_save(self):
        model = AtariConv_v6()
        import inspect
        print(inspect.getmro(AtariConv_v6))
        model.save('8834739821')
        model = Storeable.load('8834739821')
        assert model is not None


    def test_restore(self):

        r = Restoreable('one','two')
        r.metadata['fisk'] = 'frisky'
        r.save('8834739829')
        r = Storeable.load('8834739829')
        print(r.one, r.two)
        assert r is not None and r.one == 'one'
        m = Storeable.load_metadata('8834739829')
        assert m['fisk'] == 'frisky'
        print(m)

    def test_save_to_data_dir(self):
        model = AtariConv_v6()
        model.save('8834739821','c:\data')
        model = Storeable.load('8834739821','c:\data')
        assert model is not None

    def test_save_to_data_dir_random(self):
        model = AtariConv_v6([64,64,64,64,64])
        print(model)
        name = model.save(data_dir='c:\data')
        model = Storeable.load(name, data_dir='c:\data')
        print(model)
        assert model is not None

class TestModelDB(TestCase):
    def testModelDB(self):
        mdb = ModelDb('c:\data')
        mdb.print_data()

    def testTop(self):
        mdb = ModelDb('c:\data')
        top = mdb.topNLossbyModelGuid(2)
        print(top)

class TestElastic(TestCase):
    def test_connect(self):
        # connect to our cluster
        from elasticsearch import Elasticsearch
        es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    def test_sync(self):
        mdb = ModelDb('c:\data')
        mdb.sync_to_elastic()

    def test_printData(self):
        mdb = ModelDb('c:\data')
        mdb.print_data_for('V9VTQAC8LHI7K2G6')


