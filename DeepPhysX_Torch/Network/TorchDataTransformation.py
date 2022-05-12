from typing import Tuple, Optional
from torch import Tensor
from collections import namedtuple

from DeepPhysX_Core.Network.DataTransformation import DataTransformation


class TorchDataTransformation(DataTransformation):
    """
    | TorchDataTransformation is dedicated to data operations before and after network predictions.

    :param namedtuple config: Namedtuple containing the parameters of the network manager
    """

    def __init__(self, config: namedtuple):

        super().__init__(config)
        self.data_type = Tensor

    @DataTransformation.check_type
    def transform_before_prediction(self, data_in: Tensor) -> Tensor:
        """
        | Apply data operations before network's prediction.

        :param Tensor data_in: Input data
        :return: Transformed input data
        """

        return data_in

    @DataTransformation.check_type
    def transform_before_loss(self, data_out: Tensor,
                              data_gt: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        | Apply data operations between network's prediction and loss computation.

        :param Tensor data_out: Prediction data
        :param Optional[Tensor] data_gt: Ground truth data
        :return: Transformed prediction data, transformed ground truth data
        """

        return data_out, data_gt

    @DataTransformation.check_type
    def transform_before_apply(self, data_out: Tensor) -> Tensor:
        """
        | Apply data operations between loss computation and prediction apply in environment.

        :param Tensor data_out: Prediction data
        :return: Transformed prediction data
        """

        return data_out

    def __str__(self) -> str:
        """
        :return: String containing information about the TorchDataTransformation object
        """

        description = "\n"
        description += f"  {self.__class__.__name__}\n"
        description += f"    Data type: {self.data_type}\n"
        description += f"    Transformation before prediction: Identity\n"
        description += f"    Transformation before loss: Identity\n"
        description += f"    Transformation before apply: Identity\n"
        return description
