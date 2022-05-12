from typing import Any, Optional, Type, Union

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import NetworkType, OptimizationType, DataTransformationType
from DeepPhysX_Torch.Network.TorchDataTransformation import TorchDataTransformation
from DeepPhysX_Torch.Network.TorchNetwork import TorchNetwork
from DeepPhysX_Torch.Network.TorchOptimization import TorchOptimization

NetworkType = Union[NetworkType, TorchNetwork]
OptimizationType = Union[OptimizationType, TorchOptimization]
DataTransformationType = Union[DataTransformationType, TorchDataTransformation]


class TorchNetworkConfig(BaseNetworkConfig):
    """
    | TorchNetworkConfig is a configuration class to parameterize and create TorchNetwork, TorchOptimization and
      TorchDataTransformation for the NetworkManager.

    :param Type[TorchNetwork] network_class: BaseNetwork class from which an instance will be created
    :param Type[TorchOptimization] optimization_class: BaseOptimization class from which an instance will be created
    :param Type[TorchDataTransformation] data_transformation_class: DataTransformation class from which an instance will
                                                                    be created
    :param Optional[str] network_dir: Name of an existing network repository
    :param str network_name: Name of the network
    :param str network_type: Type of the network
    :param int which_network: If several networks in network_dir, load the specified one
    :param bool save_each_epoch: If True, network state will be saved at each epoch end; if False, network state
                                 will be saved at the end of the training
    :param Optional[float] lr: Learning rate
    :param bool require_training_stuff: If specified, loss and optimizer class can be not necessary for training
    :param Optional[Any] loss: Loss class
    :param Optional[Any] optimizer: Network's parameters optimizer class
    """

    def __init__(self,
                 network_class: Type[TorchNetwork] = TorchNetwork,
                 optimization_class: Type[TorchOptimization] = TorchOptimization,
                 data_transformation_class: Type[TorchDataTransformation] = TorchDataTransformation,
                 network_dir: Optional[str] = None,
                 network_name: str = 'Network',
                 network_type: str = 'TorchNetwork',
                 which_network: int = 0,
                 save_each_epoch: bool = False,
                 lr: Optional[float] = None,
                 require_training_stuff: bool = True,
                 loss: Optional[Any] = None,
                 optimizer: Optional[Any] = None):

        BaseNetworkConfig.__init__(self,
                                   network_class=network_class,
                                   optimization_class=optimization_class,
                                   data_transformation_class=data_transformation_class,
                                   network_dir=network_dir,
                                   network_name=network_name,
                                   network_type=network_type,
                                   which_network=which_network,
                                   save_each_epoch=save_each_epoch,
                                   lr=lr,
                                   require_training_stuff=require_training_stuff,
                                   loss=loss,
                                   optimizer=optimizer)

        # Change default config values for network only (configs for optimization and data_transformation are the same)
        self.network_config = self.make_config(config_name='network_config',
                                               network_name=network_name,
                                               network_type=network_type)

    def create_network(self) -> NetworkType:
        """
        | Create an instance of network_class with given parameters.

        :return: BaseNetwork object from network_class and its parameters.
        """

        return BaseNetworkConfig.create_network(self)

    def create_optimization(self) -> OptimizationType:
        """
        | Create an instance of optimization_class with given parameters.

        :return: BaseOptimization object from optimization_class and its parameters.
        """

        return BaseNetworkConfig.create_optimization(self)

    def create_data_transformation(self) -> DataTransformationType:
        """
        | Create an instance of data_transformation_class with given parameters.

        :return: DataTransformation object from data_transformation_class and its parameters.
        """

        return BaseNetworkConfig.create_data_transformation(self)
