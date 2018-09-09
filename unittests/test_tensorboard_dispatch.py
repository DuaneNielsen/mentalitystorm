from unittest import TestCase
from mentalitystorm import Dispatcher, Observable, TensorBoard, TensorBoardObservable

class Model(Dispatcher, Observable):
    def __init__(self):
        Dispatcher.__init__(self)
        self.variable = 0

    def global_step(self):
        self.updateObservers('tb_step', None, {'func':'tb_step'})

    def writeScalar(self):
        self.updateObservers('training_loss', self.variable, {'func':'tb_scalar','name':'loss/train_loss'})
        self.variable += 1

class TestTensorBoardDispatch(TestCase):
    def test_init(self):
        tb = TensorBoard('testcase')

    def test_global_step(self):

        tb = TensorBoard('testcase_gs')
        m = Model()
        m.registerView('tb_step', tb)
        m.global_step()
        assert tb.global_step == 1

    def test_scalar_update(self):
        tb = TensorBoard('testcase_scalar')
        m = Model()
        m.registerView('tb_step', tb)
        m.registerView('training_loss', tb)
        for i in range (100):
            m.writeScalar()
            m.global_step()

class Model2(Dispatcher, Observable, TensorBoardObservable):
    def __init__(self):
        Dispatcher.__init__(self)
        self.loss = 0

    def test(self):
        for _ in range(100):
            self.writeTestLossToTB(self.loss)
            self.writeTrainingLossToTB(self.loss)
            self.loss += 1
            self.tb_global_step()

class TestTBDispatchConvenianceMethods(TestCase):
    def test_convenience(self):
        tb = TensorBoard('testcase_convenience')
        m = Model2()
        tb.register(m)
        m.test()
