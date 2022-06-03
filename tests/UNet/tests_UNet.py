from unittest import TestCase
from torch import rand

from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig


class TestUNet(TestCase):

    def setUp(self):
        config = UNetConfig(input_size=[10, 10, 10],
                            nb_input_channels=1,
                            nb_output_channels=3,
                            nb_steps=3,
                            two_sublayers=True,
                            border_mode='same',
                            data_scale=10.)
        self.unet = config.create_network()
        self.transform = config.create_data_transformation()

    def test_init(self):
        # Check model
        self.assertEqual(len(self.unet.down), 4)
        self.assertEqual(len(self.unet.up), 3)
        self.assertEqual(self.unet.finalLayer.out_channels, 3)

    def test_forward(self):
        # Wrong input size
        data = rand((10, 10, 10))
        with self.assertRaises(RuntimeError):
            self.unet.forward(data)
        # Good input size
        data_t = self.transform.transform_before_prediction(data)
        self.assertEqual(self.unet.forward(data_t).shape, (1, 3, 16, 16, 16))
