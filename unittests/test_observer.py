from unittest import TestCase
from mentalitystorm import Dispatcher, Observable, View


class BooleanView(View):
    def __init__(self):
        self.state = False

    def update(self, data, metadata):
        self.state = data


class Controller(Dispatcher, Observable):
    def __init__(self):
        Dispatcher.__init__(self)

class Model(Controller, Observable):
    def __init__(self):
        Controller.__init__(self)

class Plugin(Observable):
    def call_plugin(self):
        self.updateObservers('boolean', True)

class PluggableModel(Dispatcher, Plugin):
    def __init__(self):
        Dispatcher.__init__(self)

class NonPluggableModel(Plugin):
    pass

class TestDispatcher(TestCase):

    def test_dispatch(self):
        controller = Controller()
        bv = BooleanView()
        assert not bv.state
        controller.registerView('boolean', bv)
        controller.updateObservers('boolean', True)
        assert bv.state

    def test_inheritance_parent(self):
        model = Model()
        bv = BooleanView()
        assert not bv.state
        model.registerView('boolean', bv)
        model.updateObservers('boolean', True)
        assert bv.state

    def test_inheritance_multiple(self):
        bv = BooleanView()
        assert not bv.state
        plugable = PluggableModel()
        plugable.registerView('boolean', bv)
        plugable.call_plugin()
        assert bv.state

    """ Tests that there is no exception thrown.
    """
    def test_inheritance_multiple_safe(self):

        plugable = NonPluggableModel()
        plugable.call_plugin()



