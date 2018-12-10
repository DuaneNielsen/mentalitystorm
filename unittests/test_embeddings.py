from unittest import TestCase
from mentalitystorm.data_containers import ThreeKeyEmbedding


class TestRolloutGen(TestCase):

    def test_action_embedding(self):

        fire = 1
        nop = 0
        right = 3
        left = 4

        k3 = ThreeKeyEmbedding()

        def invert(action):
            p_action = k3.toPolicy(action)
            assert k3.toEnv(p_action) == action

        invert(fire)
        invert(nop)
        invert(right)
        invert(left)
