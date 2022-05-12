import unittest
from os import devnull
from sys import stdout

from tests_FC import TestFC
from tests_FCConfig import TestFCConfig


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
