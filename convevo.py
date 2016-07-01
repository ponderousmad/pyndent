from __future__ import print_function

import copy
import itertools
import os
import lxml.etree as et

import outputer
import convnet
import mutate

def fstr(value):
    """Canonical xml float value output"""
    int_value = int(value)
    if int_value == value:
        return str(int_value)
    return str(value)

class EvoLayer(object):
    """ An evolvable layer representation.
    Each layer consists of either a conv or pool in the image section, or a hidden layer otherwise,
    followed optionally by a relu and/or dropout.
    """
    def __init__(self, primary, relu, dropout_rate, dropout_seed=None):
        self.primary = primary
        self.dropout_rate = dropout_rate
        self.dropout_seed = dropout_seed
        self.relu = relu

    def output_shape(self, input_shape):
        """Determine the output shape given the input shape."""
        return self.primary.output_shape(input_shape)

    def parameter_count(self, input_shape):
        """Get the number of model parameters in this layer."""
        return self.primary.parameter_count(input_shape)

    def reseed(self, entropy):
        """Generate new random seed values."""
        self.primary.reseed(entropy)
        self.dropout_seed = entropy.randint(1, 100000)

    def mutate(self, mutagen, is_last):
        """Randomly change layer settings."""
        self.primary.mutate(mutagen, is_last)
        self.relu = mutagen.mutate_relu(self.relu)
        self.dropout_rate = mutagen.mutate_dropout(self.dropout_rate)

    def construct(self, input_shape, operations):
        """Add the convnet operations for this layer to the list of operations."""
        for op in self.primary.construct(input_shape): 
            operations.append(op)
            op.set_number(len(operations))
        if self.relu:
            operations.append(convnet.create_relu())
        if self.dropout_rate > 0:
            operations.append(convnet.create_dropout(self.dropout_rate, self.dropout_seed))
        return self.output_shape(input_shape)

    def make_safe(self, input_shape, output_shape):
        """Adjust layer settings to conform to the input/output shape and internal constraints."""
        self.primary.make_safe(input_shape, output_shape)
        return self.output_shape(input_shape)

    def to_xml(self, parent):
        """Output the details of the layer as children of the parent node."""
        element = et.SubElement(parent, "layer")
        element.set("dropout_rate", fstr(self.dropout_rate))
        if self.dropout_seed:
            element.set("dropout_seed", str(self.dropout_seed))
        element.set("relu", str(self.relu))
        self.primary.to_xml(element)

class Initializer(object):
    """ Keep track of hyper parameters for tensor initialization"""
    def __init__(self, distribution="constant", mean=0.0, scale=1.0, seed=None):
        self.distribution = distribution
        self.mean = mean
        self.scale = scale
        self.seed = seed

    def mutate(self, mutagen):
        """Randomly alter initializer settings"""
        self.distribution = mutagen.mutate_distribution(self.distribution)
        self.mean = mutagen.mutate_initial_mean(self.mean)
        self.scale = mutagen.mutate_initial_scale(self.scale)

    def reseed(self, entropy):
        """Generate new random seed values."""
        self.seed = entropy.randint(1, 100000)

    def construct(self):
        """Set up the convnet data needed to build the TensorFlow initializer."""
        return convnet.setup_initializer(self.mean, self.scale, self.distribution, self.seed)

    def to_xml(self, parent):
        """Output the details of the initializer as children of the parent node."""
        element = et.SubElement(parent, "initializer")
        element.set("distribution", self.distribution)
        element.set("mean", fstr(self.mean))
        element.set("scale", fstr(self.scale))
        if self.seed:
            element.set("seed", str(self.seed))

class HiddenLayer(object):
    """ Non convolutional hidden layer. """
    def __init__(self, output_size, bias, initializer, l2_factor=0.0):
        self.output_size = output_size
        self.bias = bias
        self.l2_factor = l2_factor
        self.initializer = initializer

    def layer_type(self):
        """Get the type of the layer."""
        return "hidden"

    def output_shape(self, input_shape):
        """Determine the output shape given the input shape."""
        return (input_shape[0], self.output_size)

    def parameter_count(self, input_shape):
        """Get the number of model parameters in this layer."""
        count = input_shape[1] * self.output_size
        if self.bias:
            count += self.output_size
        return count

    def mutate(self, mutagen, is_last):
        """Randomly alter layer settings"""
        self.bias = mutagen.mutate_bias(self.bias)
        if not is_last:
            self.output_size = mutagen.mutate_output_size(self.output_size)
        self.initializer.mutate(mutagen)
        self.l2_factor = mutagen.mutate_l2_factor(self.l2_factor)

    def reseed(self, entropy):
        """Generate new random seed values."""
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        """Construct the convnet the operations for this layer."""
        op = convnet.create_matrix(input_shape[1], self.output_size, self.bias, self.initializer.construct())
        op.set_l2_factor(self.l2_factor)
        yield op

    def make_safe(self, input_shape, output_shape):
        """Adjust layer settings to conform to the input/output shape and internal constraints."""
        if output_shape:
            self.output_size = output_shape[-1]

    def to_xml(self, parent):
        """Output the details of the layer as children of the parent node."""
        element = et.SubElement(parent, "hidden")
        element.set("output_size", fstr(self.output_size))
        element.set("bias", str(self.bias))
        element.set("l2_factor", fstr(self.l2_factor))
        self.initializer.to_xml(element)

class ImageLayer(object):
    """Image convolution or pooling layer."""
    def __init__(self, operation, patch_size, stride, output_channels, padding, initializer, l2_factor=0.0):
        self.operation = operation
        self.patch_size = patch_size
        self.stride = stride
        self.output_channels = output_channels
        self.padding = padding
        self.l2_factor = l2_factor
        self.initializer = initializer

    def layer_type(self):
        """Get the type of the layer."""
        return "image"

    def output_shape(self, input_shape):
        """Determine the output shape given the input shape."""
        depth = self.output_channels if self.operation.startswith("conv") else input_shape[3]
        return convnet.image_output_size(
            input_shape,
            (self.patch_size, self.patch_size, input_shape[3], depth),
            (self.stride, self.stride),
            self.padding
        )

    def parameter_count(self, input_shape):
        """Get the number of model parameters in this layer."""
        if self.operation.startswith("conv"):
            count = self.patch_size * self.patch_size * input_shape[-1] * self.output_channels
            if self.operation.endswith("bias"):
                count += self.output_channels
            return count
        else:
            return 0

    def mutate(self, mutagen, is_last):
        """Randomly alter layer settings"""
        self.operation = mutagen.mutate_image_operation(self.operation)
        self.patch_size = mutagen.mutate_patch_size(self.patch_size)
        self.stride = mutagen.mutate_stride(self.stride)
        self.padding = mutagen.mutate_padding(self.padding)
        self.output_channels = mutagen.mutate_output_size(self.output_channels)
        self.initializer.mutate(mutagen)
        self.l2_factor = mutagen.mutate_l2_factor(self.l2_factor)

    def reseed(self, entropy):
        """Generate new random seed values."""
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        """Construct the convnet the operations for this layer."""
        if self.operation.startswith("conv"):
            op = convnet.create_conv(
                (self.patch_size, self.patch_size),
                (self.stride, self.stride),
                input_shape[3], self.output_channels,
                self.operation.endswith("bias"), self.padding,
                self.initializer.construct()
            )
            op.set_l2_factor(self.l2_factor)
        else:
            op = convnet.create_pool(
                self.operation,
                (self.patch_size, self.patch_size),
                (self.stride, self.stride),
                self.padding
            )
        yield op

    def make_safe(self, input_shape, output_shape):
        """Adjust layer settings to conform to the input/output shape and internal constraints."""
        self.patch_size = min(self.patch_size, input_shape[1], input_shape[2])
        self.stride = min(self.stride, self.patch_size)
        assert(not output_shape), "Image layers cannot adapt to arbitrary sizes."

    def to_xml(self, parent):
        """Output the details of the layer as children of the parent node."""
        element = et.SubElement(parent, "image")
        element.set("operation", self.operation)
        element.set("patch_size", str(self.patch_size))
        element.set("stride", str(self.stride))
        element.set("padding", self.padding)
        element.set("output_channels", str(self.output_channels))
        element.set("l2_factor", fstr(self.l2_factor))
        self.initializer.to_xml(element)

class ExpandLayer(object):
    """Image 'deconvolve' layer."""
    def __init__(self, block_size, patch_size, padding, bias, initializer, l2_factor=0.0):
        self.block_size = block_size
        self.patch_size = patch_size
        self.padding = padding
        self.bias = bias
        self.l2_factor = l2_factor
        self.initializer = initializer

    def layer_type(self):
        """Get the type of the layer."""
        return "expand"

    def output_shape(self, input_shape):
        """Determine the output shape given the input shape."""
        depth = convnet.depth_to_space_channels(input_shape[-1], self.block_size)
        shape = convnet.image_output_size(
            input_shape,
            (self.patch_size, self.patch_size, input_shape[3], depth),
            (1, 1), self.padding
        )
        return convnet.depth_to_space_shape(shape, {"block_size":self.block_size})

    def parameter_count(self, input_shape):
        """Get the number of model parameters in this layer."""
        depth = convnet.depth_to_space_channels(input_shape[-1], self.block_size)
        count = self.patch_size * self.patch_size * input_shape[-1] * depth
        if self.bias:
            count += depth 
        return count

    def mutate(self, mutagen, is_last):
        """Randomly alter layer settings"""
        self.block_size = mutagen.mutate_block_size(self.block_size)
        self.patch_size = mutagen.mutate_patch_size(self.patch_size)
        self.padding = mutagen.mutate_padding(self.padding)
        self.bias = mutagen.mutate_bias(self.bias)
        self.initializer.mutate(mutagen)
        self.l2_factor = mutagen.mutate_l2_factor(self.l2_factor)

    def reseed(self, entropy):
        """Generate new random seed values."""
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        """Construct the convnet the operations for this layer."""
        depth = convnet.depth_to_space_channels(input_shape[-1], self.block_size)
        op = convnet.create_conv(
            (self.patch_size, self.patch_size),
            (1, 1),
            input_shape[3], depth,
            self.bias, self.padding,
            self.initializer.construct()
        )
        op.set_l2_factor(self.l2_factor)
        yield op
        yield convnet.create_depth_to_space(self.block_size)

    def make_safe(self, input_shape, output_shape):
        """Adjust layer settings to conform to the input/output shape and internal constraints."""
        self.patch_size = min(self.patch_size, input_shape[1], input_shape[2])
        if output_shape:
            while True:
                shape = self.output_shape(input_shape)
                if shape[1] >= output_shape[1] and shape[2] >= output_shape[2]:
                    return
                self.block_size += 1

    def to_xml(self, parent):
        """Output the details of the layer as children of the parent node."""
        element = et.SubElement(parent, "expand")
        element.set("block_size", str(self.block_size))
        element.set("patch_size", fstr(self.patch_size))
        element.set("padding", self.padding)
        element.set("bias", str(self.bias))
        element.set("l2_factor", fstr(self.l2_factor))
        self.initializer.to_xml(element)

class Optimizer(object):
    """Optimizer and settings to use in the graph."""
    def __init__(self, name, learning_rate, alpha=None, beta=None, gamma=None, delta=None):
        self.name = name
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def mutate(self, mutagen):
        """Randomly alter optimizer settings"""
        new_name = mutagen.mutate_optimizer(self.name)
        if self.name != new_name:
            self.name = new_name
            self.default_parameters()
        self.learning_rate = mutagen.mutate_learning_rate(self.learning_rate)

    def cross(self, other, entropy):
        """Randomly combine the settings of this optimizer and the other."""
        self.learning_rate = entropy.choice([self.learning_rate, other.learning_rate])
        new_name = entropy.choice([self.name, other.name])
        if self.name != new_name:
            self.name = new_name
            self.default_parameters()
        
    def default_parameters(self):
        """Reset optimizer parameters to default values."""
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        if self.name == "GradientDescent":
            self.alpha = 0.999
            self.beta = 1000
        elif self.name == "Momentum":
            self.alpha = 1.0

    def to_xml(self, parent):
        """Output the details of the optimizer as children of the parent node."""
        element = et.SubElement(parent, "optimizer")
        element.set("name", self.name)
        element.set("learning_rate", fstr(self.learning_rate))
        if self.alpha is not None:
            element.set("alpha", fstr(self.alpha))
        if self.beta is not None:
            element.set("beta", fstr(self.beta))
        if self.gamma is not None:
            element.set("gamma", fstr(self.gamma))
        if self.delta is not None:
            element.set("delta", fstr(self.delta))
        return element

class LayerStack(object):
    """Overall structure for the network"""
    def __init__(self, flatten, optimizer=None):
        self.flatten = flatten
        self.image_layers = []
        self.expand_layers = []
        self.hidden_layers = []
        self.checkpoint_at = None
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = Optimizer("GradientDescent", 0.05)
            # Match parameters from before optimizer was in the stack.
            self.optimizer.default_parameters()

    def add_layer(self, operation, relu=False, dropout_rate=0.0, dropout_seed=None):
        """Add a new layer for the given operation."""
        layer = EvoLayer(operation, relu, dropout_rate, dropout_seed)
        if operation.layer_type() == "image":
            self.image_layers.append(layer)
        elif operation.layer_type() == "expand":
            self.expand_layers.append(layer)
        else:
            self.hidden_layers.append(layer)

    def mutate_layers(self, layer_type, layers, input_shape, output_shape, mutagen):
        """Randomly alter layer settings, duplicate, remove layers."""
        slot = mutagen.mutate_duplicate_layer(layer_type, len(layers))
        if slot is not None:
            layer = layers[slot]
            layer = copy.deepcopy(layer)
            layers.insert(slot, layer)

        layer_count = len(layers)
        if layer_type == "hidden" or (layer_type == "expand" and not self.flatten):
            layer_count -= 1
        slot = mutagen.mutate_remove_layer(layer_type, layer_count)
        if slot is not None:
            layers.pop(slot)

        shape = input_shape
        for layer in layers:
            is_last = layer == layers[-1]
            layer.mutate(mutagen, is_last)
            shape = layer.make_safe(shape, output_shape if is_last else None)
        return shape

    def mutate(self, input_shape, output_shape, mutate_options, entropy):
        """Randomly alter all aspects of the layer stack."""
        mutagen = mutate.Mutagen(mutate_options, entropy)
        self.optimizer.mutate(mutagen)
        shape = self.mutate_layers("image", self.image_layers, input_shape, None, mutagen)
        expand_output = None if self.flatten else output_shape
        shape = self.mutate_layers("expand", self.expand_layers, shape, expand_output, mutagen)

        if self.flatten:
            shape = convnet.flatten_output_shape(shape)

        return self.mutate_layers("hidden", self.hidden_layers, shape, output_shape, mutagen)

    def make_safe(self, input_shape, output_shape):
        """Try to fix any layer settings to make compatible with the specified input and output shapes."""
        shape = input_shape

        for layer in self.image_layers:
            shape = layer.make_safe(shape, None)
        for layer in self.expand_layers:
            is_output_layer = (layer == self.expand_layers[-1] and not self.flatten)
            shape = layer.make_safe(shape, output_shape if is_output_layer else None)
        if self.flatten:
            shape = convnet.flatten_output_shape(shape)
        for layer in self.hidden_layers:
            is_output_layer = (layer == self.hidden_layers[-1])
            shape = layer.make_safe(shape, output_shape if is_output_layer else None)
        return shape

    def cross(self, other, entropy):
        """Combine aspects of another layer stack into this one."""
        optimizer = copy.deepcopy(self.optimizer)
        optimizer.cross(other.optimizer, entropy)
        offspring = LayerStack(self.flatten, optimizer)
        image = mutate.cross_lists(self.image_layers, other.image_layers, entropy)
        offspring.image_layers = copy.deepcopy(image)
        expand = mutate.cross_lists(self.expand_layers, other.expand_layers, entropy)
        offspring.expand_layers = copy.deepcopy(expand)
        hidden = mutate.cross_lists(self.hidden_layers, other.hidden_layers, entropy)
        offspring.hidden_layers = copy.deepcopy(hidden)
        return offspring

    def reseed(self, entropy):
        """Generate new random seed values for all children."""
        for layer in self.image_layers:
            layer.reseed(entropy)

        for layer in self.expand_layers:
            layer.reseed(entropy)

        for layer in self.hidden_layers:
            layer.reseed(entropy)

    def construct(self, input_shape, output_shape=None):
        """Construct the convnet operations for this stack."""
        operations = []

        shape = input_shape
        for layer in self.image_layers:
            shape = layer.construct(shape, operations)

        for layer in self.expand_layers:
            shape = layer.construct(shape, operations)

        if self.flatten:
            operations.append(convnet.create_flatten())
            shape = convnet.flatten_output_shape(shape)

        for layer in self.hidden_layers:
            shape = layer.construct(shape, operations)

        if output_shape and shape != output_shape:
            if convnet.can_slice(shape, output_shape):
                operations.append(convnet.create_slice(output_shape))

        return operations

    def all_layers(self):
        """Gets all the layers."""
        return itertools.chain(self.image_layers, self.expand_layers, self.hidden_layers)

    def layer_count(self):
        """Gets a counts of all the layers."""
        return len(self.all_layers())

    def parameter_count(self, input_shape):
        count = 0
        shape = input_shape
        for layer in itertools.chain(self.image_layers, self.expand_layers):
            count += layer.parameter_count(shape)
            shape = layer.output_shape(shape)

        if self.flatten:
            shape = convnet.flatten_output_shape(shape)

        for layer in self.hidden_layers:
            count += layer.parameter_count(shape)
            shape = layer.output_shape(shape)

        return count

    def construct_optimizer(self, loss):
        """Construct the TensorFlow optimizer object."""
        options = self.optimizer;
        optimizer, step = convnet.create_optimizer(
            options.name,
            options.learning_rate,
            options.alpha,
            options.beta,
            options.gamma,
            options.delta
        )
        return optimizer.minimize(loss, global_step=step)

    def checkpoint_path(self, path=None, graph_info=None):
        """Get or set the current checkpoint path to save or restore the model."""
        if path:
            self.checkpoint_at = path
            if graph_info:
                convnet.setup_save_model(graph_info, path)
        return self.checkpoint_at

    def to_xml(self, parent = None):
        """Generate xml corresponding to this stack, optionally as a child of parent."""
        if parent is None:
            element = et.Element("evostack")
        else:
            element = et.SubElement(parent, "evostack")
        element.set("flatten", str(self.flatten))
        if self.checkpoint_at:
            element.set("checkpoint", self.checkpoint_at)
        self.optimizer.to_xml(element)
        if self.image_layers:
            children = et.SubElement(element, "layers")
            children.set("type", "image")
            for layer in self.image_layers:
                layer.to_xml(children)
        if self.expand_layers:
            children = et.SubElement(element, "layers")
            children.set("type", "expand")
            for layer in self.expand_layers:
                layer.to_xml(children)
        if self.hidden_layers:
            children = et.SubElement(element, "layers")
            children.set("type", "hidden")
            for layer in self.hidden_layers:
                layer.to_xml(children)
        return element

def serialize(stack):
    """Convert the stack to the canonical xml representation."""
    return et.tostring(stack.to_xml(), pretty_print=True)

def create_stack(convolutions, expands, flatten, hidden_sizes, init_mean, init_scale, l2, optimizer=None):
    """Construct a LayerStack instance given the specified options."""
    stack = LayerStack(flatten=flatten, optimizer=optimizer)
    default_init = lambda: Initializer("normal", mean=init_mean, scale=init_scale)

    for operation, patch_size, stride, depth, padding, relu in convolutions:
        layer = ImageLayer(operation, patch_size, stride, depth, padding, default_init(), l2_factor=l2)
        stack.add_layer(layer, relu=relu)
    for block_size, patch_size, padding, bias, relu in expands:
        layer = ExpandLayer(block_size, patch_size, padding, bias, default_init(), l2_factor=l2)
        stack.add_layer(layer, relu=relu)
    for hidden_size in hidden_sizes:
        layer = HiddenLayer(hidden_size, bias=True, initializer=default_init(), l2_factor=l2)
        stack.add_layer(layer, relu=True)

    return stack

def as_int(text, default=None, base=10):
    """Parse the specified value as an integer."""
    try:
        return int(text, base)
    except (ValueError, TypeError):
        return default

def as_float(text, default=None):
    """Parse the specified value as a floating point number."""
    try:
        return float(text)
    except (ValueError, TypeError):
        return default

def parse_initializer(operation_element):
    """Try to parse the element assuming it's an initializer."""
    init_element = operation_element.find("initializer")
    if init_element is not None:
        distribution = init_element.get("distribution")
        mean = as_float(init_element.get("mean"), 0.0)
        scale = as_float(init_element.get("scale"), 1.0)
        seed = as_int(init_element.get("seed"))
        if distribution:
            return Initializer(distribution, mean, scale, seed)
        print("Bad initializer:", et.tostring(init_element))
        return None
    print("Missing initializer element")
    return None

def parse_image(image_element):
    """Try and parse the element assuming it's an image layer operation."""
    if image_element is not None:
        operation = image_element.get("operation")
        patch_size = as_int(image_element.get("patch_size"))
        stride =  as_int(image_element.get("stride"))
        padding = image_element.get("padding")
        outputs = as_int(image_element.get("output_channels"))
        l2_factor = as_float(image_element.get("l2_factor"), 0.0)
        initializer = parse_initializer(image_element)
        if operation and patch_size and stride and outputs and padding and initializer:
            return ImageLayer(operation, patch_size, stride, outputs, padding, initializer, l2_factor)
        print("Bad image layer:", et.tostring(image_element))
        return None
    print("Missing image element.")
    return None

def parse_expand(expand_element):
    """Try and parse the element assuming it's an expand layer operation."""
    if expand_element is not None:
        block_size = as_int(expand_element.get("block_size"))
        bias = expand_element.get("bias") == "True"
        patch_size = as_int(expand_element.get("patch_size"))
        padding = expand_element.get("padding")
        l2_factor = as_float(expand_element.get("l2_factor"), 0.0)
        initializer = parse_initializer(expand_element)
        if block_size and patch_size and padding and initializer:
            return ExpandLayer(block_size, patch_size, padding, bias, initializer, l2_factor)
        print("Bad expand layer:", et.tostring(expand_element))
        return None
    print("Missing expand element.")
    return None

def parse_hidden(hidden_element):
    """Try and parse the element assuming it's a hidden layer operation."""
    if hidden_element is not None:
        outputs = as_int(hidden_element.get("output_size"))
        bias = hidden_element.get("bias") == "True"
        l2_factor = as_float(hidden_element.get("l2_factor"), 0.0)
        initializer = parse_initializer(hidden_element)
        if outputs and initializer:
            return HiddenLayer(outputs, bias, initializer, l2_factor)
        print("Bad hidden layer:", et.tostring(hidden_element))
        return None
    print("Missing hidden element.")
    return None

def parse_operation(layer_element, layer_type):
    """Try and parse the operation for a layer element given it's type."""
    if layer_type == "image":
        return parse_image(layer_element.find("image"))
    elif layer_type == "expand":
        return parse_expand(layer_element.find("expand"))
    else:
        return parse_hidden(layer_element.find("hidden"))
        
def parse_optimizer(optimizer_element):
    """Try and parse the element assuming it's an optimizer."""
    if optimizer_element is not None:
        name = optimizer_element.get("name")
        learning_rate = as_float(optimizer_element.get("learning_rate"))
        alpha = as_float(optimizer_element.get("alpha"))
        beta = as_float(optimizer_element.get("beta"))
        gamma = as_float(optimizer_element.get("gamma"))
        delta = as_float(optimizer_element.get("delta"))
        if name and learning_rate:
            return Optimizer(name, learning_rate, alpha, beta, gamma, delta)
        print("Bad optimizer.")
        return None
    print("No optimizer found.")
    return None

def parse_stack(stack_element):
    """Try and parse the element assuming it's an entire layer stack."""
    optimizer = parse_optimizer(stack_element.find("optimizer"))
    stack = LayerStack(stack_element.get("flatten") == "True", optimizer)
    checkpoint_path = stack_element.get("checkpoint")
    if checkpoint_path:
        stack.checkpoint_path(checkpoint_path)
    for layers in stack_element.iter("layers"):
        layer_type = layers.get("type")
        for layer in layers.iter("layer"):
            relu = layer.get("relu") == "True"
            dropout_rate = as_float(layer.get("dropout_rate"), 0)
            dropout_seed = as_int(layer.get("dropout_seed"))
            operation = parse_operation(layer, layer_type)
            if operation:
                stack.add_layer(operation, relu, dropout_rate, dropout_seed)
    return stack

def parse_population(population_element, include_score):
    """Try and parse the children of the population element as scored layer stacks.
    If the score is included, returns a list of (stack, score) tuples."""
    population = []
    mutate_seed = as_int(population_element.get("mutate_seed"))
    eval_seed = as_int(population_element.get("eval_seed"))
    for result in population_element.iter("result"):
        stack_element = result.find("evostack")
        stack = parse_stack(stack_element)
        if include_score:
            score = as_float(result.get("score"), -10) 
            population.append((stack, score))
        else:
            population.append(stack)
    return (population, mutate_seed, eval_seed)
    
def load_population(file, include_score=False):
    """Try and load a population of layer stacks from the specified file.
    If the score is included, returns a list of (stack, score) tuples."""
    tree = et.parse(file)
    return parse_population(tree.getroot(), include_score)

def load_stack(file):
    """Try and load a layer stack from the specified file."""
    tree = et.parse(file)
    return parse_stack(tree.getroot())

def breed(parents, options, entropy):
    """Combine one or more parents to produce an offspring and mutate it."""
    if (len(parents) < 2 or parents[0] is parents[1]):
        offspring = copy.deepcopy(parents[0])
    else:
        offspring = parents[0].cross(parents[1], entropy)
    offspring.mutate(options["input_shape"], options.get("output_shape"), options, entropy)
    offspring.reseed(entropy)
    return offspring

def rebreed(parents, options, entropy):
    """Repeated breeding for increased variation."""
    for i in xrange(entropy.randint(1, 5)):
        offspring = [
            breed(parents, options, entropy),
            breed(parents, options, entropy)
        ]
        parents = offspring
    return offspring[0]

def output_results(results, path, filename, mutate_seed=None, eval_seed=None):
    """Output a population of scored layer stacks to the specified file."""
    root = et.Element("population")
    if mutate_seed is not None:
        root.set("mutate_seed", str(mutate_seed))
    if eval_seed is not None:
        root.set("eval_seed", str(eval_seed))
    
    for stack, score in results:
        eval = et.SubElement(root, "result")
        eval.set("score", fstr(score))
        stack.to_xml(eval)

    outputer.setup_directory(path)

    with open(os.path.join(path, filename), "w") as text_file:
        text_file.write(et.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True))

