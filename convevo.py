from __future__ import print_function

import copy
import lxml.etree as et

import convnet
import mutate

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

    def to_xml(self, parent):
        element = et.SubElement(parent, "layer")
        element.set("dropout_rate", str(self.dropout_rate))
        if self.dropout_seed:
            element.set("dropout_seed", str(self.dropout_seed))
        element.set("relu", str(self.relu))
        self.primary.to_xml(element)

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

    def to_xml(self, parent):
        element = et.SubElement(parent, "initializer")
        element.set("distribution", self.distribution)
        element.set("mean", str(self.mean))
        element.set("scale", str(self.scale))
        if self.seed:
            element.set("seed", str(self.seed))

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

    def to_xml(self, parent):
        element = et.SubElement(parent, "hidden")
        element.set("output_size", str(self.output_size))
        element.set("bias", str(self.bias))
        self.initializer.to_xml(element)

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

    def to_xml(self, parent):
        element = et.SubElement(parent, "image")
        element.set("operation", self.operation)
        element.set("patch_size", str(self.patch_size))
        element.set("stride", str(self.stride))
        element.set("padding", self.padding)
        element.set("output_channels", str(self.output_channels))
        self.initializer.to_xml(element)

class LayerStack(object):
    """Overall structure for the network"""
    def __init__(self, flatten):
        self.flatten = flatten
        self.image_layers = []
        self.hidden_layers = []

    def add_layer(self, operation, relu=False, dropout_rate=0):
        layer = EvoLayer(operation, relu, dropout_rate)
        if operation.is_image():
            self.image_layers.append(layer)
        else:
            self.hidden_layers.append(layer)

    def mutate_layers(self, is_image, layers, mutagen):
        slot = mutagen.mutate_duplicate_layer(is_image, len(layers))
        if slot != None:
            layer = layers[slot]
            layers.insert(slot, copy.deepcopy(layer))
            
        slot = mutagen.mutate_remove_layer(is_image, len(layers))
        if slot != None:
            layers.pop(slot)

        for layer in layers:
            layer.mutate(mutagen)

    def mutate(self, seed):
        mutagen = mutate.Mutagen(seed)

        self.mutate_layers(True, self.image_layers, mutagen)
        self.mutate_layers(False, self.hidden_layers, mutagen)

    def reseed(self, entropy):
        for layer in self.image_layers:
            layer.reseed(entropy)

        for layer in self.hidden_layers:
            layer.mutate(entropy)

    def construct(self, input_shape):
        layers = []

        shape = input_shape
        for layer in self.image_layers:
            shape = layer.construct(shape, layers)

        if self.flatten:
            layers.append(convnet.create_flatten_layer())
            shape = convnet.flatten_output_shape(shape)


        for layer in self.hidden_layers:
            shape = layer.construct(shape, layers)

        return layers

    def to_xml(self, parent = None):
        if parent:
            element = et.SubElement(parent, "evostack")
        else:
            element = et.Element("evostack")
        element.set("flatten", str(self.flatten))
        if self.image_layers:
            children = et.SubElement(element, "layers")
            children.set("type", "image")
            for layer in self.image_layers:
                layer.to_xml(children)
        if self.image_layers:
            children = et.SubElement(element, "layers")
            children.set("type", "hidden")
            for layer in self.hidden_layers:
                layer.to_xml(children)
        return element