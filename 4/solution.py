from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            updater.inertia = updater.inertia * self.momentum + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return np.maximum(inputs, 0)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs * (self.forward_inputs >= 0)


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        return np.exp(inputs - inputs.max(axis=1, keepdims=True)) / \
                  np.exp(inputs - inputs.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        el1 = grad_outputs * self.forward_outputs
        el2 = np.sum(self.forward_outputs * grad_outputs, axis=1, keepdims=True) * self.forward_outputs
        return el1 - el2

# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return inputs @ self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        #update weights and biases gradients
        self.weights_grad = self.forward_inputs.T @ grad_outputs
        self.biases_grad = grad_outputs.sum(axis=0)

        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        print("y_gt, y_pred value")
        print(y_gt, y_pred)
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """

        return np.mean(-(y_gt * np.log(y_pred)).sum(axis=1, keepdims=True), axis= 0)

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        nz = np.copy(y_pred)
        nz[abs(y_pred) <= eps] = eps
        return np.divide(-y_gt, nz) / y_gt.shape[0]

# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.01))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(100, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 64, 10, x_valid=x_valid, y_valid=y_valid, shuffle=False)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    n, c, ih, iw = inputs.shape
    c, d, kh, kw = kernels.shape

    kernels = np.flip(kernels, axis=(-2, -1))

    oh, ow = ih - kh + 1 + 2 * padding, iw - kw + 1 + 2 * padding
    out = np.zeros((n, c, oh, ow), dtype=inputs.dtype)

    padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    for i in range(oh):
        for j in range(ow):
            a = kernels * padded_inputs[:, :, i:i + kh, j:j + kw][:, None, :, :, :]
            out[:, :, i, j] = np.sum(a, axis=(-1, -2, -3), keepdims=False).reshape((n, c))

    return out

# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernel_size) // 2

        self.biases_grad = np.mean(np.sum(inputs, axis=(2, 3)), axis=0)
        self.kernels_grad= np.zeros_like(self.kernels)
        return convolve(inputs, self.kernels, padding) + self.biases[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernel_size) // 2

        X = np.flip(self.forward_inputs, axis=(-2, -1))

        self.kernels_grad = convolve(X.swapaxes(0, 1), grad_outputs.swapaxes(0, 1),
                                     padding).swapaxes(0, 1)

        self.biases_grad = np.sum(grad_outputs, axis=(0, -2, -1))

        K = np.flip(self.kernels, axis=(-2, -1))

        result = convolve(grad_outputs, np.transpose(K, (1, 0, 2, 3)), padding)

        return result

# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        mask = np.lib.stride_tricks.sliding_window_view(inputs, window_shape=(self.pool_size, self.pool_size),
                                                              axis=(-2, -1))
        mask = mask[:, :, ::self.pool_size, ::self.pool_size]

        if self.pool_mode == 'max':
            flat_pwindows = mask.reshape(mask.shape[:4] + (-1,))
            self.forward_idxs = np.argmax(flat_pwindows, axis=4, keepdims=True)

            return mask.max(axis=(-1, -2))
        elif self.pool_mode == 'avg':
            return mask.mean(axis=(-1, -2))

    def compute_grad_avg(self):
        grad_avg = np.ones(self.forward_inputs.shape) / (self.pool_size * self.pool_size)
        return grad_avg

    def compute_grad_max(self):
        grad_max = np.zeros(self.forward_idxs.shape[:4] + (self.pool_size * self.pool_size,))
        np.put_along_axis(grad_max, self.forward_idxs, 1, axis=-1)
        grad_max = grad_max.reshape(self.forward_idxs.shape[:4] + (self.pool_size, self.pool_size,))
        grad_max = np.moveaxis(grad_max, -2, -3).reshape(self.forward_inputs.shape)
        return grad_max

    def backward_impl(self, grad_outputs):

        if self.pool_mode == 'max':
            grads = self.compute_grad_max()
        elif self.pool_mode == 'avg':
            grads = self.compute_grad_avg()

        grad_tiled = np.tile(grad_outputs[..., None, None], (1, 1, 1, 1, self.pool_size, self.pool_size))
        grad_tiled = np.moveaxis(grad_tiled, -2, -3).reshape(self.forward_inputs.shape)

        return grads * grad_tiled


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            var = np.var(inputs, axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.reshape(-1,)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.reshape(-1,)
            self.forward_centered_inputs = inputs - mean
            self.forward_inverse_std = 1 / np.sqrt(var + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
        else:
            self.forward_inverse_std = 1 / np.sqrt(eps + self.running_var)[None, :, None, None]
            self.forward_centered_inputs = (inputs - self.running_mean[None, :, None, None])
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
        return self.forward_normalized_inputs * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.koef = grad_outputs * self.gamma[None, :, None, None] * self.forward_inverse_std
        n, d, input_h, input_w = self.forward_centered_inputs.shape

        dvar = (-1 / 2 * (self.gamma[None, :, None, None] * grad_outputs * self.forward_centered_inputs * self.forward_inverse_std ** 3)).sum(
            axis=(0, 2, 3), keepdims=True)
        dmean = (-self.gamma[None, :, None, None] * grad_outputs * self.forward_inverse_std).sum(axis=(0, 2, 3),
                                                                        keepdims=True) - 2 * dvar * self.forward_centered_inputs.sum(
                                                                        axis=(0, 2, 3), keepdims=True) / input_h / input_w / n

        final = self.koef + dvar * 2 * self.forward_centered_inputs / input_h / input_w / n + dmean / input_h / input_w / n

        return final

        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(self.forward_inputs.shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.uniform(size=inputs.shape) > self.p
            return inputs * self.forward_mask
        else:
            return inputs * (1 - self.p)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.01, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(16, kernel_size=3, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(pool_size=2, pool_mode='max'))
    model.add(Conv2D(32, kernel_size=3))
    model.add(Conv2D(32, kernel_size=3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(pool_size=2, pool_mode='max'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dropout(p = 0.2))
    model.add(Dense(10))
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 64, 10, x_valid=x_valid, y_valid=y_valid, shuffle=False)

    # your code here /\
    return model

# ============================================================================
