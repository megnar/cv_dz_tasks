import numpy as np

def calculate_shift(i1, i2):
    corr = np.fft.ifft2(np.fft.fft2(i1) * np.conj(np.fft.fft2(i2)))
    return np.unravel_index(np.argmax(corr), corr.shape)


def shift_2d_array(array, shift_u, shift_v):
    array = np.roll(array, shift_u, axis=0)
    array = np.roll(array, shift_v, axis=1)
    return array


def align(img, g_abs):
    img = (img * 255).astype(np.uint8)

    frame = img.shape[0] // 3

    B, G, R = img[:frame, :], img[frame:2 * frame, :], img[2 * frame:3 * frame, :]

    new_w = int(B.shape[0] * 0.05)
    new_h = int(B.shape[1] * 0.05)
    new_g = np.array([g_abs[0] - frame - new_w, g_abs[1] - new_h])
    shape = np.array(B.shape) - np.array([2 * new_w, 2 * new_h])
    B = B[new_w: B.shape[0] - new_w, new_h: B.shape[1] - new_h]
    G = G[new_w: G.shape[0] - new_w, new_h: G.shape[1] - new_h]
    R = R[new_w: R.shape[0] - new_w, new_h: R.shape[1] - new_h]

    B_shift, R_shift = calculate_shift(G, B), calculate_shift(G, R)
    G_B = (new_g + B_shift) % shape - new_g
    G_R = (new_g + R_shift) % shape - new_g

    b_abs = g_abs[0] - frame - G_B[0], g_abs[1] - G_B[1]
    r_abs = g_abs[0] + frame - G_R[0], g_abs[1] - G_R[1]

    B_new = shift_2d_array(B, G_B[0], G_B[1])
    R_new = shift_2d_array(R, G_R[0], G_B[1])

    return np.dstack((R_new, G, B_new)), b_abs, r_abs