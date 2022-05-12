from unittest import TestCase

from DeepPhysX_Torch.UNet.UNetConfig import UNetConfig


class TestUNetConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            UNetConfig(input_size={})
            UNetConfig(nb_dims=0.)
            UNetConfig(nb_input_channels=0.)
            UNetConfig(nb_output_channels=0.)
            UNetConfig(nb_first_layer_channels=0.)
            UNetConfig(two_sublayers='True')
            UNetConfig(skip_merge='True')
            UNetConfig(border_mode=0)
            UNetConfig(data_scale=0)
        # ValueError
        with self.assertRaises(ValueError):
            UNetConfig(nb_dims=1)
            UNetConfig(nb_input_channels=0)
            UNetConfig(nb_output_channels=0)
            UNetConfig(nb_first_layer_channels=0)
            UNetConfig(border_mode='')
        # Network config
        unet_config = UNetConfig()
        self.assertTrue('network_name' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.network_name, 'UNetNetwork')
        self.assertTrue('network_type' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.network_type, 'UNet')
        self.assertTrue('nb_dims' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.nb_dims, 3)
        self.assertTrue('nb_input_channels' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.nb_input_channels, 1)
        self.assertTrue('nb_first_layer_channels' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.nb_first_layer_channels, 64)
        self.assertTrue('nb_output_channels' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.nb_output_channels, 3)
        self.assertTrue('nb_steps' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.nb_steps, 3)
        self.assertTrue('two_sublayers' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.two_sublayers, True)
        self.assertTrue('border_mode' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.border_mode, 'valid')
        self.assertTrue('skip_merge' in unet_config.network_config._fields)
        self.assertEqual(unet_config.network_config.skip_merge, False)
        # DataTransformation config
        self.assertTrue('input_size' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.input_size, [0, 0, 0])
        self.assertTrue('nb_input_channels' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.nb_input_channels, 1)
        self.assertTrue('nb_output_channels' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.nb_output_channels, 3)
        self.assertTrue('nb_steps' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.nb_steps, 3)
        self.assertTrue('two_sublayers' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.two_sublayers, True)
        self.assertTrue('border_mode' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.border_mode, 'valid')
        self.assertTrue('data_scale' in unet_config.data_transformation_config._fields)
        self.assertEqual(unet_config.data_transformation_config.data_scale, 1.)
