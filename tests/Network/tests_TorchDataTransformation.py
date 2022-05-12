from unittest import TestCase
from torch import Tensor, rand
from numpy.random import random

from DeepPhysX_Torch.Network.TorchNetworkConfig import TorchNetworkConfig


class TestTorchDataTransformation(TestCase):

    def setUp(self):
        self.transform = TorchNetworkConfig().create_data_transformation()

    def test_init(self):
        # Default values
        self.assertEqual(self.transform.data_type, Tensor)

    def test_check_type(self):
        # TypeError
        data = random((10, 3))
        with self.assertRaises(TypeError):
            self.transform.transform_before_prediction(data)
            self.transform.transform_before_loss(data, data)
            self.transform.transform_before_apply(data)

    def test_transform_before_prediction(self):
        # Identity
        data = rand((10, 3))
        self.assertTrue(bool((self.transform.transform_before_prediction(data) == data).all()))

    def test_transform_before_loss(self):
        # Identity
        data = rand((10, 3))
        self.assertTrue(bool((self.transform.transform_before_loss(data)[0] == data).all()))

    def test_transform_before_apply(self):
        # Identity
        data = rand((10, 3))
        self.assertTrue(bool((self.transform.transform_before_apply(data) == data).all()))
