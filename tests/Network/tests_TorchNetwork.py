from unittest import TestCase
from torch import Tensor
from torch.cuda import is_available
import os
from numpy import array, float32, ndarray
from numpy.random import random

from DeepPhysX_Torch.Network.TorchNetworkConfig import TorchNetworkConfig


class TestTorchNetwork(TestCase):

    def setUp(self):
        self.network = TorchNetworkConfig().create_network()

    def test_init(self):
        # Default values
        self.assertEqual(self.network.device, None)
        self.assertEqual(self.network.config.network_name, 'TorchNetwork')
        self.assertEqual(self.network.config.network_type, 'TorchNetwork')

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.network.predict(None)
            self.network.forward(None)

    def test_set_mode(self):
        # Check model training mode
        self.network.set_train()
        self.assertTrue(self.network.training)
        # Check model eval mode
        self.network.set_eval()
        self.assertFalse(self.network.training)

    def test_set_device(self):
        # Check model device
        self.network.set_device()
        device = 'cuda' if is_available() else 'cpu'
        self.assertEqual(self.network.device.type, device)

    def test_parameters_methods(self):
        # Check model is empty
        self.assertEqual(self.network.nb_parameters(), 0)
        # Check save and load parameters
        init_state = self.network.get_parameters()
        self.network.save_parameters('network')
        self.network.load_parameters('network.pth')
        self.assertEqual(self.network.get_parameters(), init_state)
        os.remove('network.pth')

    def tests_tensor_conversion(self):
        # Conversion from numpy to tensor
        n_data = array(random((2, 1, 3)), dtype=float32).round(2)
        t_data_no_grad = self.network.transform_from_numpy(n_data, grad=False)
        self.assertEqual(type(t_data_no_grad), Tensor)
        self.assertFalse(t_data_no_grad.requires_grad)
        t_data = self.network.transform_from_numpy(n_data)
        self.assertTrue(t_data.requires_grad)
        # Conversion from tensor to numpy
        r_data = self.network.transform_to_numpy(t_data)
        self.assertEqual(type(r_data), ndarray)
        self.assertTrue((r_data == n_data).all())
