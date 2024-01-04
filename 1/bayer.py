import numpy as np
from copy import deepcopy
import math
from scipy.signal import convolve2d


def get_bayer_masks(n_rows, n_cols):
    first_type = np.tile(np.array([1, 0], 'bool'), (n_cols + 1) // 2)[:n_cols]
    second_type = np.tile(np.array([0, 1], 'bool'), (n_cols + 1) // 2)[:n_cols]
    third_type = np.tile(np.array([0, 0], 'bool'), (n_cols + 1) // 2)[:n_cols]
    red_matr = np.tile([second_type, third_type], ((n_rows + 1) // 2, 1))[:n_rows]
    green_matr = np.tile([first_type, second_type], ((n_rows + 1) // 2, 1))[:n_rows]
    blue_matr = np.tile([third_type, first_type], ((n_rows + 1) // 2, 1))[:n_rows]
    result = np.dstack((red_matr, green_matr, blue_matr))
    return result

def get_colored_img(raw_img):
    narray = np.array(raw_img)
    mask = get_bayer_masks(narray.shape[0], narray.shape[1])
    red = np.where(mask[..., 0], narray, 0)
    green = np.where(mask[..., 1], narray, 0)
    blue = np.where(mask[..., 2], narray, 0)
    return np.dstack((red, green, blue))

def simple_interpolation(channel, num):
    channel = channel.astype(np.uint16)
    x, y = channel.shape

    kernel = np.ones((3, 3))

    result = convolve2d(channel, kernel, mode='same', boundary='fill', fillvalue=0)

    for i in range(x):
        for j in range(y):
            total = 4
            if num % 2 == 0 and (i + j) % 2 == 0:
                total = 2

            result[i][j] = result[i][j] / total

    for i in range(x):
        for j in range(y):
            if num == 0:
                if i % 2 == 0 and j % 2 == 1:
                    result[i][j] = channel[i][j]
            elif num == 2:
                if i % 2 == 1 and j % 2 == 0:
                    result[i][j] = channel[i][j]
            else:
                if (i + j) % 2 == 0:
                    result[i][j] = channel[i][j]
    return result.astype(np.uint8)


def bilinear_interpolation(colored_img):
    red = simple_interpolation(colored_img[..., 0], 0)
    green = simple_interpolation(colored_img[..., 1], 1)
    blue = simple_interpolation(colored_img[..., 2], 2)
    return np.dstack((red, green, blue))

def improved_interpolation(raw_img):
    channel = raw_img.astype(np.int_)

    x, y = channel.shape

    # fill all kernels

    kernelr2g = np.array([[0, 0, -1, 0, 0],
                          [0, 0, 2, 0, 0],
                          [-1, 2, 4, 2, -1],
                          [0, 0, 2, 0, 0],
                          [0, 0, -1, 0, 0]], np.int_) * 2
    kernelr2g = np.flip(kernelr2g)

    kernelr2b = np.array([[0, 0, -3, 0, 0],
                          [0, 4, 0, 4, 0],
                          [-3, 0, 12, 0, -3],
                          [0, 4, 0, 4, 0],
                          [0, 0, -3, 0, 0]], np.int_)
    kernelr2b = np.flip(kernelr2b)

    kernelg2r_row = np.array([[0, 0, 1, 0, 0],
                              [0, -2, 0, -2, 0],
                              [-2, 8, 10, 8, -2],
                              [0, -2, 0, -2, 0],
                              [0, 0, 1, 0, 0]], np.int_)
    kernelg2r_row = np.flip(kernelg2r_row)

    kernelg2r_col = np.array([[0, 0, -2, 0, 0],
                              [0, -2, 8, -2, 0],
                              [1, 0, 10, 0, 1],
                              [0, -2, 8, -2, 0],
                              [0, 0, -2, 0, 0]], np.int_)
    kernelg2r_col = np.flip(kernelg2r_col)

    r2g = convolve2d(channel, kernelr2g, mode='same', boundary='fill', fillvalue=0) / 16
    r2b = convolve2d(channel, kernelr2b, mode='same', boundary='fill', fillvalue=0) / 16
    g2r_row = convolve2d(channel, kernelg2r_row, mode='same', boundary='fill', fillvalue=0) / 16
    g2r_col = convolve2d(channel, kernelg2r_col, mode='same', boundary='fill', fillvalue=0) / 16

    red = np.zeros((x, y)).astype(np.int_)
    green = np.zeros((x, y)).astype(np.int_)
    blue = np.zeros((x, y)).astype(np.int_)

    for i in range(x):
        for j in range(y):
            if i % 2 == 0 and j % 2 == 1:
                num2 = 0
            elif i % 2 == 1 and j % 2 == 0:
                num2 = 2
            else:
                num2 = 1

            # fill red

            if num2 == 0:
                red[i][j] = channel[i][j]
            elif num2 == 2:
                red[i][j] = r2b[i][j]
            elif num2 == 1 and i % 2 == 0:
                red[i][j] = g2r_row[i][j]
            else:
                red[i][j] = g2r_col[i][j]

            # fill blue

            if num2 == 2:
                blue[i][j] = channel[i][j]
            elif num2 == 0:
                blue[i][j] = r2b[i][j]
            elif num2 == 1 and i % 2 == 1:
                blue[i][j] = g2r_row[i][j]
            else:
                blue[i][j] = g2r_col[i][j]

            # fill green

            if num2 == 1:
                green[i][j] = channel[i][j]
            else:
                green[i][j] = r2g[i][j]

    r = slice(2, -2), slice(2, -2)

    gt_img = np.dstack((red, green, blue))

    return np.clip(gt_img, 0, 255)

def MSE(img_pred, img_gt):
    diff = np.sum((img_pred - img_gt) ** 2)
    for i in img_pred.shape:
        diff = diff / i
    return diff


def compute_psnr(img_pred, img_gt):
    img_gt = img_gt.astype(np.float64)
    img_pred = img_pred.astype(np.float64)
    mse = MSE(img_pred, img_gt)
    if mse == 0:
        raise ValueError
    return 10 * math.log(np.max(img_gt) ** 2 / mse, 10)
