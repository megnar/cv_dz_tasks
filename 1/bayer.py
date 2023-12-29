import pytest
import numpy as np
#from numpy import tile, dstack

def get_bayer_masks(n_rows, n_cols):
    first_type = np.tile(np.array([1, 0], 'bool'), (n_cols + 1) // 2)
    second_type = np.tile(np.array([0, 1], 'bool'), (n_cols + 1) // 2)
    third_type = np.tile(np.array([0, 0], 'bool'), (n_cols + 1) // 2)
    red_matr = np.tile([second_type, third_type], (n_rows + 1) // 2)
    green_matr = np.tile([first_type, second_type], (n_rows + 1) // 2)
    blue_matr = np.tile([third_type, first_type], (n_rows + 1) // 2)
    result = np.dstack((red_matr, green_matr, blue_matr))
    return result
