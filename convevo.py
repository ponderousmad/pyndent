from __future__ import print_function

import convnet

class EvoLayer(object):
    """ An evolvable layer representation.
    Each layer consists of either a conv or pool in the image section, or a hidden layer otherwise,
    followed optionally by a relu and/or dropout.
    """
    def __init__(self, primary, relu, dropout_rate):
        self.primary = primary
        self.dropout_rate = dropout_rate
        self.dropout_seed = None # Use graph level seed.
        self.relu = False

    def output_shape(self, input_shape):
        return self.primary.output_shape(input_shape)

    def reseed(self, entropy):
        self.primary.reseed(entropy)
        self.dropout_seed = entropy.randint(1, 100000)

    def mutate(self, mutagen):
        self.primary.mutate(mutagen)
        self.relu = mutagen.mutate_relu(self.relu)
        self.dropout_rate = mutagen.mutate_dropout(self.dropout_rate)

    def construct(self, input_shape, layers):
        layers.append(self.primary.construct(input_shape))
        if self.relu:
            layers.append(convnet.create_relu_layer())
        if self.dropout_rate > 0:
            layers.append(convnet.create_dropout_layer(self.dropout_rate, self.dropout_seed))
        return self.output_shape(input_shape)

class Initializer(object):
    """ Keep track of hyper parameters for tensor initialization"""
    def __init__(self, distribution="constant", mean=0.0, scale=1.0):
        self.distribution = distribution
        self.mean = mean
        self.scale = scale
        self.seed = None

    def mutate(self, mutagen):
        self.distribution = mutagen.mutate_distribution(self.distribution)
        self.mean = mutagen.mutate_initial_mean(self.mean)
        self.scale = mutagen.mutate_initial_scale(self.scale)

    def reseed(self, entropy):
        self.seed = entropy.randint(1, 100000)

    def construct(self):
        return convnet.setup_initializer(self.mean, self.scale, self.distribution, self.seed)

class HiddenLayer(object):
    """ Non convolutional hidden layer. """
    def __init__(self, output_size, bias, initializer):
        self.output_size = output_size
        self.bias = bias
        self.initializer = initializer

    def is_image(self):
        return False

    def output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    def mutate(self, mutagen):
        self.output_size = mutagen.mutate_output_size(self.output_size)
        self.initializer.mutate(mutagen)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        return convnet.create_matrix_layer(input_shape[1], self.output_size, self.bias, self.initializer.construct())

class ImageLayer(object):
    def __init__(self, operation, patch_size, stride, output_channels, padding, initializer):
        self.operation = operation
        self.patch_size = patch_size
        self.stride = stride
        self.output_channels = output_channels
        self.padding = padding
        self.initializer = initializer

    def is_image(self):
        return True

    def output_shape(self, input_shape):
        return convnet.image_output_size(
            input_shape,
            (self.patch_size, self.patch_size, input_shape[3], self.output_channels),
            (self.stride, self.stride),
            self.padding
        )

    def mutate(self, mutagen):
        self.operation = mutagen.mutate_image_operation(self.operation)
        self.patch_size = mutagen.mutate_patch_size(self.patch_size)
        self.stride = mutagen.mutate_stride(self.stride)
        self.padding = mutagen.mutate_padding(self.padding)
        self.output_channels = mutagen.mutate_output_size(self.output_channels)
        self.initializer.mutate(mutagen)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        if self.operation.startswith("conv"):
            return convnet.create_conv_layer(
                (self.patch_size, self.patch_size),
                (self.stride, self.stride),
                input_shape[3], self.output_channels,
                self.operation.endswith("bias"), self.padding
            )
        else:
            return convnet.create_pool_layer(
                self.operation,
                (self.patch_size, self.patch_size)
                (self.stride, self.stride),
                self.padding
            )

class LayerStack(object):
    """Overall structure for the network"""
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.image_layers = []
        self.hidden_layers = []

    def add_layer(self, operation, relu=False, dropout_rate=0):
        layer = EvoLayer(operation, relu, dropout_rate)
        if operation.is_image():
            self.image_layers.append(layer)
        else:
            self.hidden_layers.append(layer)

    def mutate(self, seed):
        muatagen = mutate.Mutagen(seed)

        for layer in self.image_layers:
            layer.mutate(mutagen)

        for layer in self.hidden_layers:
            layer.mutate(mutagen)

    def reseed(self, entropy):
        for layer in self.image_layers:
            layer.reseed(entropy)

        for layer in self.hidden_layers:
            layer.mutate(entropy)

    def construct(self):
        layers = []

        shape = self.input_size
        for layer in self.image_layers:
            shape = layer.construct(shape, layers)

        layers.append(convnet.create_flatten_layer())
        shape = convnet.flatten_output_shape(shape)

        for layer in self.hidden_layers:
            shape = layer.construct(shape, layers)

        return layers
