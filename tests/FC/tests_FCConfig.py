from unittest import TestCase

from DeepPhysX_Torch.FC.FCConfig import FCConfig


class TestFCConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            FCConfig(dim_output=0.5)
            FCConfig(dim_layers=())
        # Network config
        fc_config = FCConfig()
        self.assertTrue('network_name' in fc_config.network_config._fields)
        self.assertEqual(fc_config.network_config.network_name, 'FCNetwork')
        self.assertTrue('network_type' in fc_config.network_config._fields)
        self.assertEqual(fc_config.network_config.network_type, 'FC')
        self.assertTrue('dim_output' in fc_config.network_config._fields)
        self.assertEqual(fc_config.network_config.dim_output, 0)
        self.assertTrue('dim_layers' in fc_config.network_config._fields)
        self.assertEqual(fc_config.network_config.dim_layers, [])
