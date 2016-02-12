__author__ = 'ori22_000'

import unittest
from unittest import TestCase
from OriKerasExtension.MyNPHelper import myMatrixConcat

import numpy as np


class TestMyMatrixConcat(TestCase):
    def test_myMatrixConcat(self):
        list_a = [[1, 2], [3, 4]]
        list_b = [[5, 6], [7, 8]]

        target_array = np.asarray([[1, 2],
                                   [3, 4],
                                   [5, 6],
                                   [7, 8]]);
        return_val = myMatrixConcat(list_a, list_b)
        np.testing.assert_array_equal(return_val, target_array)

    def test_myMatrixConcatWithNone(self):
        list_a = None
        list_b = [[5, 6], [7, 8]]

        target_array = np.asarray([[5, 6],
                                   [7, 8]]);
        return_val = myMatrixConcat(list_a, list_b)
        np.testing.assert_array_equal(return_val, target_array)


if __name__ == '__main__':
    unittest.main()
