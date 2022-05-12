from typing import Dict, Any
from torch import Tensor
from collections import namedtuple

from DeepPhysX_Core.Network.BaseOptimization import BaseOptimization


class TorchOptimization(BaseOptimization):
    """
    | TorchOptimization is dedicated to network optimization: compute loss between prediction and target, update
      network parameters.

    :param namedtuple config: Namedtuple containing TorchOptimization parameters
    """

    def __init__(self, config: namedtuple):

        BaseOptimization.__init__(self, config)

    def set_loss(self) -> None:
        """
        | Initialize the loss function.
        """

        if self.loss_class is not None:
            self.loss = self.loss_class()

    def compute_loss(self, prediction: Tensor, ground_truth: Tensor, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute loss from prediction / ground truth.

        :param Tensor prediction: Tensor produced by the forward pass of the Network
        :param Tensor ground_truth: Ground truth tensor to be compared with prediction
        :param Dict[str, Any] data: Additional data sent as dict to compute loss value
        :return: Loss value
        """

        self.loss_value = self.loss(prediction, ground_truth)
        return self.transform_loss(data)

    def transform_loss(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        | Apply a transformation on the loss value using the potential additional data.

        :param Dict[str, Any] data: Additional data sent as dict to compute loss value
        :return: Transformed loss value
        """

        return {'loss': self.loss_value.item()}

    def set_optimizer(self, net) -> None:
        """
        | Define an optimization process.

        :param BaseNetwork net: Network whose parameters will be optimized.
        """

        if (self.optimizer_class is not None) and (self.lr is not None):
            self.optimizer = self.optimizer_class(net.parameters(), self.lr)

    def optimize(self) -> None:
        """
        | Run an optimization step.
        """

        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def __str__(self) -> str:
        """
        :return: String containing information about the TorchOptimization object
        """

        return BaseOptimization.__str__(self)
