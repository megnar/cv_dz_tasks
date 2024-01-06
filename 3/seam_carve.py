from copy import deepcopy
import numpy as np
from scipy.signal import correlate2d
from scipy import ndimage


def rgb2yuv(img):
    transformation_matrix = np.array([[0.299, 0.587, 0.114],
                                      [-0.14713, -0.28886, 0.436],
                                      [0.615, -0.51499, -0.10001]])
    img_yuv = np.dot(img, transformation_matrix.T)
    return img_yuv


def find_grad(img):
    matr_x = np.array([[-1, 0, 1]])
    matr_y = np.array([[-1], [0], [1]])

    grad_x = correlate2d(img, matr_x, mode="same", boundary="symm")
    grad_y = correlate2d(img, matr_y, mode="same", boundary="symm")
    grad_x[:, 1:-1] /= 2
    grad_y[1:-1, :] /= 2
    grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return grad_norm


def compute_energy(img):
    img_yuv = rgb2yuv(img)
    grad = find_grad(img_yuv[..., 0])
    return grad

def compute_seam_matrix(energy, mode, mask = None):
    result = deepcopy(energy).astype(np.float64)
    x, y = energy.shape
    if mask is not None:
        mask = mask.astype(np.float64)
        result += mask * 256 * x * y
    if mode == "horizontal":
        for i in range(1, x):
            arr = deepcopy(result[i - 1])
            kernel_size = (3)
            min_arr = ndimage.minimum_filter(arr, size=kernel_size)
            result[i] = result[i] + min_arr

    elif mode == 'vertical':
        for i in range(1, y):
            arr = deepcopy(result[..., i - 1])
            kernel_size = (3)
            min_arr = ndimage.minimum_filter(arr, size=kernel_size)
            result[..., i] = result[..., i] + min_arr
    return result

def horizontal_remove_seam(array, mask):
    x, y = array.shape
    result = np.zeros((x, y - 1))
    for i in range(x):
        result[i] = array[i][mask[i] == 0]
    return result

def vertical_remove_seam(array, mask):
    x, y = array.shape
    result = np.zeros((x - 1, y))
    for i in range(y):
        result[..., i] = array[..., i][mask[..., i] == 0]
    return result

def remove_minimal_seam(image, seam_matrix, mode, my_mask=None):
    x, y, _ = image.shape
    mask = np.zeros((x, y), np.uint8)

    if mode == "horizontal shrink":
        val, index = seam_matrix[x - 1][0], 0
        for j in range(y):
            if seam_matrix[x - 1][j] < val:
                val = seam_matrix[x - 1][j]
                index = j
        mask[x - 1][index] = 1
        for i in reversed(range(1, x)):
            val2, index2 = seam_matrix[i - 1][max(0, index - 1)], max(0, index - 1)
            for j in range(max(0, index - 1), min(index + 2, y)):
                if seam_matrix[i - 1][j] < val2:
                    val2 = seam_matrix[i - 1][j]
                    index2 = j
            mask[i - 1][index2] = 1
            index = index2

    if mode == "vertical shrink":
        val, index = seam_matrix[0][y - 1], 0
        for j in range(x):
            if seam_matrix[j][y - 1] < val:
                val = seam_matrix[j][y - 1]
                index = j
        mask[index][y - 1] = 1
        for i in reversed(range(1, y)):
            val2, index2 = seam_matrix[max(0, index - 1)][i - 1], max(0, index - 1)
            for j in range(max(0, index - 1), min(index + 2, x)):
                if seam_matrix[j][i - 1] < val2:
                    val2 = seam_matrix[j][i - 1]
                    index2 = j
            mask[index2][i - 1] = 1
            index = index2

    new_img = None

    if mode == "horizontal shrink":
        new_img = np.zeros((x, y - 1, 3), dtype= np.uint8)
        if my_mask is not None:
            my_mask = horizontal_remove_seam(my_mask, mask)
        for channel in range(image.shape[2]):
            array = image[:, :, channel]
            new_img[:, :, channel] =  horizontal_remove_seam(array.astype(np.uint8), mask)

    elif mode == "vertical shrink":
        new_img = np.zeros((x - 1, y, 3), dtype= np.uint8)
        if my_mask is not None:
            my_mask = vertical_remove_seam(my_mask, mask)
        for channel in range(image.shape[2]):
            array = np.array(image[:, :, channel])
            new_img[:, :, channel] = vertical_remove_seam(array.astype(np.uint8), mask)

    return (new_img, my_mask, mask)

def seam_carve(img, mode, mask = None):
    energy = compute_energy(img)
    seam_matrix = compute_seam_matrix(energy, mode.split()[0], mask)
    return remove_minimal_seam(img, seam_matrix, mode, mask)


