from typing import Dict
import torch
from numpy import ndarray
from gc import collect as gc_collect
from psutil import cpu_count
from collections import namedtuple

from DeepPhysX.Core.Network.BaseNetwork import BaseNetwork


class TorchNetwork(torch.nn.Module, BaseNetwork):
    """
    | TorchNetwork is a network class to compute predictions from input data according to actual state.

    :param namedtuple config: Namedtuple containing BaseNetwork parameters
    """

    def __init__(self, config: namedtuple):

        torch.nn.Module.__init__(self)
        BaseNetwork.__init__(self, config)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        | Gives input_data as raw input to the neural network.

        :param torch.Tensor input_data: Input tensor
        :return: Network prediction
        """

        raise NotImplementedError

    def set_train(self) -> None:
        """
        | Set the Network in train mode (compute gradient).
        """

        self.train()

    def set_eval(self) -> None:
        """
         | Set the Network in eval mode (does not compute gradient).
         """

        self.eval()

    def set_device(self) -> None:
        """
        | Set computer device on which Network's parameters will be stored and tensors will be computed.
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Garbage collector run
            gc_collect()
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(cpu_count(logical=True) - 1)
        self.to(self.device)
        print(f"[{self.__class__.__name__}]: Device is {self.device}")

    def load_parameters(self, path: str) -> None:
        """
        | Load network parameter from path.

        :param str path: Path to Network parameters to load
        """

        self.load_state_dict(torch.load(path, map_location=self.device))

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        | Return the current state of Network parameters.

        :return: Network parameters
        """

        return self.state_dict()

    def save_parameters(self, path: str) -> None:
        """
        | Saves the network parameters to the path location.

        :param str path: Path where to save the parameters.
        """

        path = path + '.pth'
        torch.save(self.state_dict(), path)

    def nb_parameters(self) -> int:
        """
        | Return the number of parameters of the network.

        :return: Number of parameters
        """

        return sum(p.numel() for p in self.parameters())

    def transform_from_numpy(self, data: ndarray, grad: bool = True) -> torch.Tensor:
        """
        | Transform and cast data from numpy to the desired tensor type.

        :param ndarray data: Array data to convert
        :param bool grad: If True, gradient will record operations on this tensor
        :return: Converted tensor
        """

        data = torch.as_tensor(data.astype(self.config.data_type), device=self.device)
        if grad:
            data.requires_grad_()
        return data

    def transform_to_numpy(self, data: torch.Tensor) -> ndarray:
        """
        | Transform and cast data from tensor type to numpy.

        :param torch.Tensor data: Any to convert
        :return: Converted array
        """

        return data.cpu().detach().numpy().astype(self.config.data_type)

    @staticmethod
    def print_architecture(architecture) -> str:
        """
        Format the network architecture string description.

        :return: String containing the network architecture description
        """

        lines = architecture.splitlines()
        architecture = ''
        for line in lines:
            architecture += '\n      ' + line
        return architecture

    def __str__(self):
        """
        :return: String containing information about the BaseNetwork object
        """

        description = BaseNetwork.__str__(self)
        description += f"    Device: {self.device}\n"
        return description
