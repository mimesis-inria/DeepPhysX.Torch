import unittest
from os import devnull
from sys import stdout

from tests_UNet import TestUNet
from tests_UNetConfig import TestUNetConfig
from tests_UNetDataTransformation import TestUNetDataTransformation


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
