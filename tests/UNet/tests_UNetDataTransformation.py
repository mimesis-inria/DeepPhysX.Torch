from unittest import TestCase
from torch import Tensor, rand, ones

from DeepPhysX_Torch.UNet.UNetConfig import UNetConfig


class TestUNetDataTransformation(TestCase):

    def setUp(self):
        self.transform = UNetConfig(input_size=[10, 10, 10],
                                    nb_input_channels=1,
                                    nb_output_channels=3,
                                    nb_steps=3,
                                    two_sublayers=True,
                                    border_mode='same',
                                    data_scale=10.).create_data_transformation()

    def test_init(self):
        transform = UNetConfig().create_data_transformation()
        # Default values
        self.assertEqual(transform.data_type, Tensor)
        self.assertEqual(transform.input_size, [0, 0, 0])
        self.assertEqual(transform.nb_steps, 3)
        self.assertEqual(transform.nb_input_channels, 1)
        self.assertEqual(transform.nb_output_channels, 3)
        self.assertEqual(transform.data_scale, 1.)
        self.assertEqual(transform.config.two_sublayers, True)
        self.assertEqual(transform.config.border_mode, 'valid')

    def test_transform_before_prediction(self):
        # Compute transformation
        data = rand((10, 10, 10))
        data_t = self.transform.transform_before_prediction(data)
        # Check shape and norm
        self.assertEqual(data_t.shape, (1, 1, 16, 16, 16))
        self.assertEqual(data_t.norm(), data.norm())

    def test_transform_before_loss(self):
        # Ensure transform_before_prediction has been called before
        _ = self.transform.transform_before_prediction(rand((10, 10, 10)))
        # Compute transformation
        data_out, data_gt = rand((1, 1, 16, 16, 16)), rand((10, 10, 30))
        data_out_t, data_gt_t = self.transform.transform_before_loss(data_out, data_gt)
        # Check only out shape (inverse padding will lead to norm change)
        self.assertEqual(data_out_t.shape, (1, 10, 10, 10, 1))
        # Check gt shape and norm
        self.assertEqual(data_gt_t.shape, (1, 10, 10, 10, 3))
        self.assertEqual(data_gt_t.norm(), (self.transform.data_scale * data_gt).norm())

    def test_transform_before_apply(self):
        # Compute transformation
        data = rand((10, 10, 30))
        data_t = self.transform.transform_before_apply(data)
        # Check shape and norm
        self.assertEqual(data_t.shape, data.shape)
        self.assertEqual(data_t.norm(), (data / self.transform.data_scale).norm())
