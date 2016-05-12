from __future__ import print_function

import tensorflow as tf

# Node/Layer Types:
# * Matrix
#   * Dimensions (height, width, depth)
# * Relu
# * Dropout
#   * Rate
# * Conv
#   * Dimensions (height, width, channels)
#   * Stride (height, width)
#   * Padding Type (same, valid)
# * Pool
#   * Type (max, avg)
#   * Size (height, width)
#   * Stride (height, width)


# Parameter setup functions

def no_parameters(options):
    return ()

def setup_matrix(options):
    initialize_matrix = options["init"]
    size = options["size"]
    matrix = tf.Variable(initialize_matrix(size))
    if options["bias"]:
        initialize_bias = options["bias_init"]
        bias = tf.Variable(initialize_bias(size[-1:]))
        return (matrix, bias)
    return (matrix,)

def setup_initializer(mean=0.0, scale=1.0, distribution="constant", seed=None):
    return {
        "constant":  lambda shape: tf.fill(shape, mean),
        "uniform":   lambda shape: tf.random_uniform(shape, mean - scale, mean + scale, seed=seed),
        "normal":    lambda shape: tf.random_normal(shape, mean, scale, seed=seed),
        "truncated": lambda shape: tf.truncated_normal(shape, mean, scale, seed=seed)
    }[distribution]

# Shape functions.

def same_output_shape(input_shape, options=None):
    return input_shape

def matrix_output_shape(input_shape, options):
    matrix_size = options["size"]
    return (int(input_shape[0]), matrix_size[1])

def flatten_output_shape(input_shape, options=None):
    return (int(input_shape[0]), int(input_shape[1] * input_shape[2] * input_shape[3]))

def unflatten_output_shape(input_shape, options):
    size = options["size"]
    pixels = size[0] * size[1]
    return (int(input_shape[0]), size[0], size[1], int(input_shape[1] / pixels))

def image_output_size(input_shape, size, stride, padding):
    if len(size) > 2 and input_shape[3] != size[2]:
        print("Matrix size incompatible!")

    height = size[0]
    width  = size[1]
    out_depth = size[3] if len(size) > 2 else int(input_shape[3])

    input_height = input_shape[1]
    input_width  = input_shape[2]

    if padding == "VALID":
        input_height -= height - 1
        input_width  -= width - 1

    return (
        int(input_shape[0]),
        (input_height + stride[0] - 1) / stride[0],
        (input_width  + stride[1] - 1) / stride[1],
        out_depth
    )

def image_size_options(options):
    return {key: options[key] for key in ('size', 'stride', 'padding')}

def image_output_shape(input_shape, options):
    return image_output_size(input_shape, **image_size_options(options))

# Node connection functions

def apply_matrix(input_node, train, parameters, options):
    application = tf.matmul(input_node, parameters[0])
    if len(parameters) > 1:
        return application + parameters[1]
    return application

def apply_relu(input_node, train, parameters, options):
    return tf.nn.relu(input_node)

def apply_dropout(input_node, train, parameters, options):
    if train:
        return tf.nn.dropout(input_node, options["dropout_rate"], seed=options["seed"])
    else:
        return input_node

def apply_conv(input_node, train, parameters, options):
    stride = options["stride"]
    output = tf.nn.conv2d(input_node, parameters[0], [1, stride[0], stride[1], 1], padding=options["padding"])

    if options["bias"]:
        output = output + parameters[1]

    return output

def apply_pool(input_node, train, parameters, options):
    if options["pool_type"].startswith("max"):
        pool_function = tf.nn.max_pool
    else:
        pool_function = tf.nn.avg_pool
    stride = [1, options["stride"][0], options["stride"][1], 1]
    size = [1, options["size"][0], options["size"][1], 1]
    return pool_function(input_node, size, stride, padding=options["padding"])

def apply_flatten(input_node, train, parameters, options):
    return tf.reshape(input_node, flatten_output_shape(input_node.get_shape(), options))

def apply_unflatten(input_node, train, parameters, options):
    return tf.reshape(input_node, unflatten_output_shape(input_node.get_shape(), options))

def shape_test(shape, options, func):
    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=shape)
        parameters = setup_matrix(options)
        result = func(input, False, parameters, options)
        return tuple(int(d) for d in result.get_shape())

class Layer(object):
    """Setup and keep track of graph parameters and nodes for a layer."""
    def __init__(self, options, parameter_setup, node_setup):
        self.options = options
        self.parameter_setup = parameter_setup
        self.node_setup = node_setup
        self.parameters = None
        self.loss_factor = 0

    def setup_parameters(self):
        self.parameters = self.parameter_setup(self.options)
    
    def set_l2_factor(self, loss_factor):
        self.loss_factor = loss_factor
    
    def update_loss(self, loss):
        if self.loss_factor:
            return loss + self.loss_factor * tf.nn.l2_loss(self.parameters[0])
        return loss       

    def connect(self, input_node, train):
        node = self.node_setup(input_node, train, self.parameters, self.options)
        return node

# Layer setup functions.

def create_matrix_layer(inputs, outputs, bias=True, init=setup_initializer(distribution="normal", scale=0.1)):
    size = (inputs, outputs)
    options = {
        "size": size,
        "bias": bias,
        "init": lambda size: init(size),
        "bias_init": lambda size: init((outputs,))
    }
    return Layer(options, setup_matrix, apply_matrix)

def create_relu_layer():
    return Layer({}, no_parameters, apply_relu)

def create_dropout_layer(rate, seed):
    options = {
        "dropout_rate": rate,
        "seed": seed
    }
    return Layer(options, no_parameters, apply_dropout)

def create_conv_layer(patch_size, stride, in_channels, out_channels, bias=True, padding="SAME", init=setup_initializer(distribution="normal", scale=0.1)):
    options = {
        "size": patch_size + (in_channels, out_channels),
        "bias": bias,
        "init": init,
        "bias_init": init,
        "stride": stride,
        "padding": padding
    }
    return Layer(options, setup_matrix, apply_conv)

def create_pool_layer(strategy, patch_size, stride, padding="SAME"):
    options = {
        "pool_type": strategy,
        "size": patch_size,
        "stride": stride,
        "padding": padding
    }
    return Layer(options, no_parameters, apply_pool)

def create_flatten_layer():
    options = {}
    return Layer(options, no_parameters, apply_flatten)

def create_unflatten_layer(size):
    options = {
        "size": size
    }
    return Layer(options, no_parameters, apply_unflatten)

