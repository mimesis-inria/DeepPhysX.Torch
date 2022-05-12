from unittest import TestCase

from DeepPhysX_Torch.Network.TorchNetworkConfig import TorchNetworkConfig


class TestTorchOptimization(TestCase):

    def setUp(self):
        self.optimization = TorchNetworkConfig().create_optimization()

    def test_init(self):
        # Default values
        self.assertEqual(self.optimization.manager, None)
        self.assertEqual(self.optimization.loss_class, None)
        self.assertEqual(self.optimization.loss_value, 0.)
        self.assertEqual(self.optimization.loss, None)
        self.assertEqual(self.optimization.optimizer_class, None)
        self.assertEqual(self.optimization.optimizer, None)
        self.assertEqual(self.optimization.lr, None)
