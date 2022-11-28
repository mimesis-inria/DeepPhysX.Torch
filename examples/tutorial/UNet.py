"""
UNet.py
How to configure, create and use the UNet Architecture.
"""

# Python related
import torch
from time import time

# DeepPhysX's PyTorch imports
from DeepPhysX.Torch.UNet.UNetConfig import UNetConfig


def main():

    # UNet configuration
    unet_config = UNetConfig(network_dir=None,                      # Path with a trained network to load parameters
                             network_name="MyUnet",                 # Nickname of the network
                             which_network=0,                       # Instance index in case several where saved
                             save_each_epoch=False,                 # Save network parameters at each epoch end or not
                             lr=1e-5,                               # Leaning rate
                             require_training_stuff=True,           # Loss & optimizer are required or not for training
                             loss=torch.nn.MSELoss,                 # Loss class to use
                             optimizer=torch.optim.Adam,            # Optimizer class to manage the learning process
                             input_size=[5, 10, 10],                # Input size used to compute data transformations
                             nb_dims=3,                             # Number of dimensions of the input
                             nb_input_channels=1,                   # Number of channels of the input
                             nb_first_layer_channels=128,           # Number of channels after the first layer
                             nb_output_channels=1,                  # Number of channels of the output
                             nb_steps=3,                            # Number of steps on each U side of the network
                             two_sublayers=True,                    # Define if each UNet layer is duplicated or not
                             border_mode='same',                    # Zero padding mode between 'valid'=0 & 'same'=1
                             skip_merge=False,                      # Outputs must be merged on same U levels or not
                             data_scale=1.)                         # Scale to apply for data transformations

    """
    The following methods are automatically called by the NetworkManager in a normal DeepPhysX pipeline.
    They are only used here to demonstrate what is performed during the pipeline.
    """

    # Creating network, data_transformation and network_optimization
    print("Creating UNet...")
    unet = unet_config.create_network()
    unet.set_device()
    unet.set_eval()
    data_transformation = unet_config.create_data_transformation()
    optimization = unet_config.create_optimization()
    print("\nNETWORK DESCRIPTION:", unet)
    print("\nDATA TRANSFORMATION DESCRIPTION:", data_transformation)
    print("\nOPTIMIZATION DESCRIPTION:", optimization)

    # Data transformations and forward pass of Unet on a random tensor
    t = torch.rand((1, 500), dtype=torch.float, device=unet.device)
    data = {'input': t}
    start_time = time()
    unet_input = data_transformation.transform_before_prediction(data)
    unet_output = unet.predict(unet_input)
    unet_loss, _ = data_transformation.transform_before_loss(unet_output, None)
    unet_pred = data_transformation.transform_before_apply(unet_loss)['prediction']
    unet_apply = unet_pred.reshape(t.shape)
    end_time = time()
    print(f"Prediction time: {round(end_time - start_time, 5) * 1e3} ms")
    print("Tensor shape:", t.shape)
    print("Input shape:", unet_input['input'].shape)
    print("Output shape:", unet_output['prediction'].shape)
    print("Loss shape:", unet_loss['prediction'].shape)
    print("Prediction shape:", unet_pred.shape)
    print("Apply shape:", unet_apply.shape)


if __name__ == '__main__':
    main()
