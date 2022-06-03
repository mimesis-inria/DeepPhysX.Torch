from unittest import TestCase
import os

from DeepPhysX.Torch.Network.TorchNetworkConfig import TorchNetworkConfig, TorchNetwork, TorchOptimization, TorchDataTransformation


class TestTorchNetworkConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            TorchNetworkConfig(network_dir=0)
            TorchNetworkConfig(network_name=0)
            TorchNetworkConfig(which_network="0")
            TorchNetworkConfig(save_each_epoch=-1)
        # ValueError
        with self.assertRaises(ValueError):
            TorchNetworkConfig(network_dir=os.path.join(os.getcwd(), 'network'))
            TorchNetworkConfig(which_network=-1)
        # Default values
        network_config = TorchNetworkConfig()
        self.assertEqual(network_config.network_class, TorchNetwork)
        self.assertEqual(network_config.optimization_class, TorchOptimization)
        self.assertEqual(network_config.data_transformation_class, TorchDataTransformation)
        self.assertEqual(network_config.network_dir, None)
        self.assertEqual(network_config.training_stuff, False)
        self.assertEqual(network_config.which_network, 0)
        self.assertEqual(network_config.save_each_epoch, False)
        # Network config
        self.assertTrue('network_name' in network_config.network_config._fields)
        self.assertEqual(network_config.network_config.network_name, 'TorchNetwork')
        self.assertTrue('network_type' in network_config.network_config._fields)
        self.assertEqual(network_config.network_config.network_type, 'TorchNetwork')
        # Optimization config
        self.assertTrue('loss' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.loss, None)
        self.assertTrue('lr' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.lr, None)
        self.assertTrue('optimizer' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.optimizer, None)

    def test_create_network(self):
        # ValueError
        self.assertRaises(ValueError, TorchNetworkConfig(network_class=Test1).create_network)
        # TypeError
        self.assertRaises(TypeError, TorchNetworkConfig(network_class=Test2).create_network)
        # No error
        self.assertIsInstance(TorchNetworkConfig().create_network(), TorchNetwork)

    def test_create_optimization(self):
        # ValueError
        self.assertRaises(ValueError, TorchNetworkConfig(optimization_class=Test1).create_optimization)
        # TypeError
        self.assertRaises(TypeError, TorchNetworkConfig(optimization_class=Test2).create_optimization)
        # No error
        self.assertIsInstance(TorchNetworkConfig().create_optimization(), TorchOptimization)

    def test_create_data_transformation(self):
        # ValueError
        self.assertRaises(ValueError, TorchNetworkConfig(data_transformation_class=Test1).create_data_transformation)
        # TypeError
        self.assertRaises(TypeError, TorchNetworkConfig(data_transformation_class=Test2).create_data_transformation)
        # No error
        self.assertIsInstance(TorchNetworkConfig().create_data_transformation(), TorchDataTransformation)


class Test1:
    def __init__(self):
        pass


class Test2:
    def __init__(self, config):
        pass
