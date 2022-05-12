import unittest
from os import devnull
from sys import stdout

from tests_TorchNetworkConfig import TestTorchNetworkConfig
from tests_TorchNetwork import TestTorchNetwork
from tests_TorchOptimization import TestTorchOptimization
from tests_TorchDataTransformation import TestTorchDataTransformation


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
