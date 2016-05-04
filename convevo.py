from __future__ import print_function

import convnet

class EvoLayer(object):
    """ An evolvable layer representation.
    Each layer consists of either a conv or pool in the image section, or a hidden layer otherwise,
    followed optionally by a relu and/or dropout.
    """
    def __init__(self, primary):
        self.primary = primary
        self.dropout_rate = 0
        self.dropout_seed = None # Use graph level seed.
        self.relu = False

    def output_size(self, input_size):
        return self.primary.output_size(input_size)

    def reseed(self, entropy):
        self.primary.reseed(entropy)
        self.dropout_seed = entropy.randint(1, 100000)

    def mutate(self, mutagen):
        self.primary.mutate(mutagen)
        self.relu = mutagen.mutate_relu(self.relu)
        self.dropout_rate = mutagen.mutate_dropout(self.dropout_rate)

    def construct(self, input_size, layers):
        layers.append(self.primary.construct(input_size))
        if self.relu:
            layers.append(convnet.create_relu_layer())
        if self.dropout_rate > 0:
            layers.append(convnet.create_dropout_layer(self.dropout_rate, self.dropout_seed))

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

    def output_size(self, input_size):
        return self.output_size

    def mutate(self, mutagen):
        self.output_size = mutagen.mutate_output_size(self.output_size)
        self.initializer.mutate(mutagen)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_size):
        return convnet.create_matrix_layer(input_size, output_size, self.bias, self.initializer.construct())

class ImageLayer(object):
    def __init__(self, operation, patch_size, stride, output_channels, padding, initializer):
        self.operation = operation
        self.patch_size = patch_size
        self.stride = stride
        self.output_channels
        self.padding = padding
        self.initializer = initializer

    def output_size(self, input_size):
        return self.output_size

    def mutate(self, mutagen):
        self.operation = mutagen.mutate_image_operation(self.operation)
        self.patch_size = mutagen.mutate_patch_size(self.patch_size)
        self.stride = mutagen.mutate_stride(self.stride)
        self.padding = mutagen.mutate_padding(self.padding)
        self.output_channels = mutagen.mutate_output_size(self.output_channels)
        self.initializer.mutate(mutagen)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_size):
        if self.operation.startswith("conv"):
            return create_conv_layer(
                (self.patch_size, self.patch_size),
                (self.stride, self.stride),
                input_size[3], self.output_channels,
                self.operation.endswith("bias"), self.padding
            )
        else:
            return create_pool_layer(
                self.operation,
                (self.patch_size, self.patch_size)
                (self.stride, self.stride),
                self.padding
            )


class LayerStack(object):
    """Overall structure for the network"""
    def __init__(self, input_size, flatten, output_size):
        self.input_size = input_size
        self.flatten = flatten
        self.output_size = output_size
        self.layers = []
        
    def mutate(self):
        pass
