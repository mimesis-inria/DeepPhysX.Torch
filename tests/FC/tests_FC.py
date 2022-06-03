from unittest import TestCase
from torch import rand

from DeepPhysX.Torch.FC.FCConfig import FCConfig, FC


class TestFC(TestCase):

    def setUp(self):
        self.fc = FCConfig(dim_layers=[10, 10, 10], dim_output=2).create_network()

    def test_init(self):
        # Check model
        self.assertEqual(len(self.fc.layers), 2 * 2)
        self.assertEqual(self.fc.config.dim_output, 2)

    def test_forward(self):
        # Wrong input size
        data = rand(10)
        with self.assertRaises(RuntimeError):
            self.fc.forward(data)
        # Good input size
        data = rand((1, 5, 2))
        self.assertEqual(self.fc.forward(data).shape, (1, 5, 2))

