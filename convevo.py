from __future__ import print_function

import copy
import datetime
import os
import lxml.etree as et

import convnet
import mutate

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
        return self.primary.output_shape(input_shape)

    def reseed(self, entropy):
        self.primary.reseed(entropy)
        self.dropout_seed = entropy.randint(1, 100000)

    def mutate(self, mutagen, is_last):
        self.primary.mutate(mutagen, is_last)
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
    def __init__(self, distribution="constant", mean=0.0, scale=1.0, seed=None):
        self.distribution = distribution
        self.mean = mean
        self.scale = scale
        self.seed = seed

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
    def __init__(self, output_size, bias, initializer, l2_factor=0.0):
        self.output_size = output_size
        self.bias = bias
        self.l2_factor = l2_factor
        self.initializer = initializer

    def is_image(self):
        return False

    def output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    def mutate(self, mutagen, is_last):
        if not is_last:
            self.output_size = mutagen.mutate_output_size(self.output_size)
        self.initializer.mutate(mutagen)
        self.l2_factor = mutagen.mutate_l2_factor(self.l2_factor)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        layer = convnet.create_matrix_layer(input_shape[1], self.output_size, self.bias, self.initializer.construct())
        layer.set_l2_factor(self.l2_factor)
        return layer

    def to_xml(self, parent):
        element = et.SubElement(parent, "hidden")
        element.set("output_size", str(self.output_size))
        element.set("bias", str(self.bias))
        element.set("l2_factor", str(self.l2_factor))
        self.initializer.to_xml(element)

class ImageLayer(object):
    def __init__(self, operation, patch_size, stride, output_channels, padding, initializer, l2_factor=0.0):
        self.operation = operation
        self.patch_size = patch_size
        self.stride = stride
        self.output_channels = output_channels
        self.padding = padding
        self.l2_factor = l2_factor
        self.initializer = initializer

    def is_image(self):
        return True

    def output_shape(self, input_shape):
        depth = self.output_channels if self.operation.startswith("conv") else input_shape[3]
        return convnet.image_output_size(
            input_shape,
            (self.patch_size, self.patch_size, input_shape[3], depth),
            (self.stride, self.stride),
            self.padding
        )

    def mutate(self, mutagen, is_last):
        self.operation = mutagen.mutate_image_operation(self.operation)
        self.patch_size = mutagen.mutate_patch_size(self.patch_size)
        self.stride = mutagen.mutate_stride(self.stride)
        self.padding = mutagen.mutate_padding(self.padding)
        self.output_channels = mutagen.mutate_output_size(self.output_channels)
        self.initializer.mutate(mutagen)
        self.l2_factor = mutagen.mutate_l2_factor(self.l2_factor)

    def reseed(self, entropy):
        self.initializer.reseed(entropy)

    def construct(self, input_shape):
        if self.operation.startswith("conv"):
            layer = convnet.create_conv_layer(
                (self.patch_size, self.patch_size),
                (self.stride, self.stride),
                input_shape[3], self.output_channels,
                self.operation.endswith("bias"), self.padding,
                self.initializer.construct()
            )
            layer.set_l2_factor(self.l2_factor)
            return layer
        else:
            return convnet.create_pool_layer(
                self.operation,
                (self.patch_size, self.patch_size),
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
        element.set("l2_factor", str(self.l2_factor))
        self.initializer.to_xml(element)

class Optimizer(object):
    """Optimizer and settings to use in the graph."""
    def __init__(self, name, alpha=None, beta=None, gamma=None, delta=None, epoch=None):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epoch = epoch

    def to_xml(self, parent):
        element = et.SubElement(parent, "optimizer")
        element.set("name", self.name)
        if self.alpha is not None:
            element.set("alpha", str(self.alpha))
        if self.beta is not None:
            element.set("beta", str(self.beta))
        if self.gamma is not None:
            element.set("gamma", str(self.gamma))
        if self.delta is not None:
            element.set("delta", str(self.delta))
        if self.epoch is not None:
            element.set("epoch", str(self.epoch))
        return element

class LayerStack(object):
    """Overall structure for the network"""
    def __init__(self, flatten, optimizer=None):
        self.flatten = flatten
        self.image_layers = []
        self.hidden_layers = []
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = Optimizer("GradientDescent", 0.05, 1.0, epoch=1000)

    def add_layer(self, operation, relu=False, dropout_rate=0.0, dropout_seed=None):
        layer = EvoLayer(operation, relu, dropout_rate, dropout_seed)
        if operation.is_image():
            self.image_layers.append(layer)
        else:
            self.hidden_layers.append(layer)

    def mutate_layers(self, is_image, layers, mutagen):
        slot = mutagen.mutate_duplicate_layer(is_image, len(layers))
        if slot is not None:
            layer = layers[slot]            
            layers.insert(slot, copy.deepcopy(layer))
        
        layer_count = len(layers)
        if not is_image:
            layer_count -= 1
        slot = mutagen.mutate_remove_layer(is_image, layer_count)
        if slot is not None:
            layers.pop(slot)

        for layer in layers:
            layer.mutate(mutagen, layer == layers[-1])

    def mutate(self, entropy):
        mutagen = mutate.Mutagen(entropy)

        self.mutate_layers(True, self.image_layers, mutagen)
        self.mutate_layers(False, self.hidden_layers, mutagen)
        
    def cross(self, other, entropy):
        offspring = LayerStack(self.flatten)
        image = mutate.cross_lists(self.image_layers, other.image_layers, entropy)
        offspring.image_layers = copy.deepcopy(image)
        hidden = mutate.cross_lists(self.hidden_layers, other.hidden_layers, entropy)
        offspring.hidden_layers = copy.deepcopy(hidden)
        return offspring

    def reseed(self, entropy):
        for layer in self.image_layers:
            layer.reseed(entropy)

        for layer in self.hidden_layers:
            layer.reseed(entropy)

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

    def construct_optimizer(self, global_step):
        optimizer = self.optimizer;
        return convnet.create_optimizer(
            optimizer.name,
            global_step,
            optimizer.alpha,
            optimizer.beta,
            optimizer.gamma,
            optimizer.delta,
            optimizer.epoch
        )

    def to_xml(self, parent = None):
        if parent is None:
            element = et.Element("evostack")
        else:
            element = et.SubElement(parent, "evostack")
        element.set("flatten", str(self.flatten))
        self.optimizer.to_xml(element)
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

    def same_as(self, other):
        selfie = et.tostring(self.to_xml())
        return selfie == et.tostring(other.to_xml())

def serialize(stack):
    return et.tostring(stack.to_xml(), pretty_print=True)

def as_int(text, default=None, base=10):
    try:
        return int(text, base)
    except ValueError:
        return default

def as_float(text, default=None):
    try:
        return float(text)
    except ValueError:
        return default

def parse_initializer(operation_element):
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

def parse_hidden(hidden_element):
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

def parse_operation(layer_element, is_image):
    if is_image:
        return parse_image(layer_element.find("image"))
    else:
        return parse_hidden(layer_element.find("hidden"))
        
def parse_optimizer(optimizer_element):
    if optimizer_element is not None:
        name = operation_element.get("name")
        alpha = as_float(operation_element.get("alpha"))
        beta = as_float(operation_element.get("beta"))
        gamma = as_float(operation_element.get("gamma"))
        delta = as_float(operation_element.get("delta"))
        epoch = as_int(operation_element.get("epoch"))
        if name:
            return Optimizer(name, alpha, beta, gamma, delta, epoch)
        print("Missing optimizer name.")
        return None
    print("No optimizer found.")
    return None

def parse_stack(stack_element):
    optimizer = parse_optimizer(stack_element.find("optimizer"))
    stack = LayerStack(stack_element.get("flatten") == "True", optimizer)
    for layers in stack_element.iter("layers"):
        is_image = (layers.get("type") == "image")
        for layer in layers.iter("layer"):
            relu = layer.get("relu") == "True"
            dropout_rate = as_float(layer.get("dropout_rate"), 0)
            dropout_seed = as_int(layer.get("dropout_seed"))
            operation = parse_operation(layer, is_image)
            if operation:
                stack.add_layer(operation, relu, dropout_rate, dropout_seed)
    return stack

def parse_population(population_element):
    population = []
    mutate_seed = as_int(population_element.get("mutate_seed"))
    eval_seed = as_int(population_element.get("eval_seed"))
    for result in population_element.iter("result"):
        stack_element = result.find("evostack")
        population.append(parse_stack(stack_element))
    return (population, mutate_seed, eval_seed)
    
def load_population(file):
    tree = et.parse(file)
    return parse_population(tree.getroot())

def breed(parents, entropy):
    if (len(parents) < 2 or parents[0] is parents[1]):
        offspring = copy.deepcopy(parents[0])
    else:
        offspring = parents[0].cross(parents[1], entropy)
    offspring.mutate(entropy)
    return offspring

def init_population(prototypes, population_target, entropy):
    population = list(prototypes)

    while (len(population) < population_target):
        parent = entropy.choice(population)
        offspring = breed([parent], entropy)
        if parent.same_as(offspring):
            print("Offspring is clone.")
        else:
            population.append(offspring)
    return population

def output_results(results, path, filename=None, mutate_seed=None, eval_seed=None):
    if not filename:
        filename = datetime.datetime.now().strftime("%Y-%m-%d~%H_%M_%S_%f")[0:-3] + ".xml"
    root = et.Element("population")
    if mutate_seed is not None:
        root.set("mutate_seed", str(mutate_seed))
    if eval_seed is not None:
        root.set("eval_seed", str(eval_seed))
    
    for stack, score in results:
        eval = et.SubElement(root, "result")
        eval.set("score", str(score))
        stack.to_xml(eval)

    try:    
        os.makedirs(path)
    except OSError:
        pass

    with open(os.path.join(path, filename), "w") as text_file:
        text_file.write(et.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True))
        
def output_error(stack, error_data, path, filename=None):
    if not filename:
        filename = datetime.datetime.now().strftime("ERR~%Y-%m-%d~%H_%M_%S_%f")[0:-3] + ".txt"

    try:    
        os.makedirs(path)
    except OSError:
        pass

    with open(os.path.join(path, filename), "w") as text_file:
        text_file.write(et.tostring(stack.to_xml(), pretty_print=True))        
        text_file.write("\n------------------------------------------------------------\n")
        for line in error_data:
            text_file.write(line)
