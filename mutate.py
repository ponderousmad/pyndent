import random

class Mutagen(object):
    """Keep track of probabilities and randomness for mutating conv nets."""
    def __init__(self, options, entropy, using_GPU=True):
        self.toggle_relu = 0.1
        self.toggle_bias = 0.05
        self.change_dropout_rate = 0.1
        self.DROPOUT_GRANULARITY = 4
        self.output_size_factors = [
            (0.04, 0.5),
            (0.02, 0.75),
            (0.02, 0.9),
            (0.02, 1.1),
            (0.02, 1.25),
            (0.04, 2.0)
        ]
        self.distributions = [
            (0.01, "constant"),
            (0.01, "uniform"),
            (0.04, "normal"),
            (0.02, "truncated")
        ]
        self.initial_means = [
            (0.03, 0),
            (0.02, 1),
            (0.01, -1)
        ]
        self.image_operations = [
            (0.08, "conv_bias"),
            (0.02, "conv"),
            (0.05, "max_pool"),
            (0.05, "avg_pool")
        ]
        self.patches = [
            (0.010, 1),
            (0.015, 2),
            (0.018, 3),
            (0.020, 4),
            (0.021, 5),
            (0.020, 6),
            (0.018, 7),
            (0.015, 8),
            (0.010, 9),
            (0.008, 10),
            (0.006, 11),
            (0.004, 12),
            (0.003, 13),
            (0.002, 14),
            (0.001, 15)
        ]
        self.strides = [
            (0.020, 1),
            (0.040, 2),
            (0.010, 3),
            (0.005, 4),
            (0.002, 5)
        ]
        self.paddings = [
            (0.15, "SAME"),
            (0.15, "VALID")
        ]
        self.block_sizes = [
            (0.10, 2),
            (0.08, 3),
            (0.05, 4),
            (0.02, 5)
        ]
        self.l2_factors = [
            (0.08, 0),
            (0.03, 0.001),
            (0.02, 0.01),
            (0.01, 0.1)
        ]
        self.add_layer = {
            "image": 0.20,
            "hidden": 0.20,
            "expand": 0.05
        }
        self.remove_layer = {
            "image": 0.15,
            "hidden": 0.15,
            "expand": 0.03
        }
        self.optimizers = [
            (0.080, "GradientDescent"),
            (0.080, "Adadelta"),
            (0.080, "Adagrad"),
            (0.001, "Momentum"),
            (0.080, "Adam"),
            (0.080, "RMSProp")
        ]
        if not using_GPU:
            self.optimizers.append((0.01, "Ftrl"))
        
        self.learning_rate_factors = [
            (0.04, 0.5),
            (0.08, 0.75),
            (0.08, 1.5),
            (0.04, 2.0)
        ]
        if options:
            if options.get("fixed_stride"):
                self.strides = []
            if options.get("fixed_padding"):
                self.paddings = []

        self.entropy = entropy

    def branch(self, bias):
        """Make a biased coin flip."""
        return self.entropy.random() < bias

    def select(self, choices, default=None):
        """Choose a new value given choices with probabilities, or return the default."""
        value = self.entropy.random()
        threshold = 0
        for entry in choices:
            threshold += entry[0]
            if value < threshold:
                return entry[1]
        return default

    def mutate_relu(self, relu):
        return (not relu) if self.branch(self.toggle_relu) else relu

    def mutate_bias(self, bias):
        return (not bias) if self.branch(self.toggle_bias) else bias

    def mutate_dropout(self, rate):
        if self.branch(self.change_dropout_rate):
            return (1.0 / self.DROPOUT_GRANULARITY) * self.entropy.randint(0, self.DROPOUT_GRANULARITY - 1)
        return rate

    def mutate_output_size(self, output_size):
        return int(output_size * self.select(self.output_size_factors, 1))

    def mutate_distribution(self, distribution):
        return self.select(self.distributions, distribution)

    def mutate_initial_mean(self, mean):
        return self.select(self.initial_means, mean)

    def mutate_initial_scale(self, scale):
        return scale * self.select(self.output_size_factors, 1)

    def mutate_image_operation(self, operation):
        return self.select(self.image_operations, operation)

    def mutate_patch_size(self, patch_size):
        return self.select(self.patches, patch_size)

    def mutate_block_size(self, block_size):
        return self.select(self.block_sizes, block_size)
        
    def mutate_stride(self, stride):
        return self.select(self.strides, stride)

    def mutate_padding(self, padding):
        return self.select(self.paddings, padding)

    def mutate_l2_factor(self, l2_factor):
        return self.select(self.l2_factors, l2_factor)

    def mutate_duplicate_layer(self, layer_type, layer_count):
        probability = self.add_layer[layer_type]
        if layer_count > 0 and self.branch(probability):
            return self.entropy.randint(0, layer_count - 1)
        return None

    def mutate_remove_layer(self, layer_type, layer_count):
        probability = self.remove_layer[layer_type]
        if layer_count > 1 and self.branch(probability):
            return self.entropy.randint(0, layer_count - 1)
        return None
    
    def mutate_optimizer(self, optimizer_name):
        return self.select(self.optimizers, optimizer_name)
        
    def mutate_learning_rate(self, rate):
        return rate * self.select(self.learning_rate_factors, 1)

def cross_lists(list_a, list_b, entropy):
    """Combine some of the start of the first list with some of the end of the second."""
    if len(list_a) < 1:
        return list_b
    if len(list_b) < 1:
        return list_a
    split_at = entropy.randint(0, len(list_a) - 1)
    result = list_b[:split_at]
    result.extend(list_a[-(len(list_a) - split_at):])
    return result

def fisher_yates_shuffle(items, entropy):
    """Based on https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle"""
    i = len(items)
    while i > 1:
        i = i - 1
        j = entropy.randint(0, i)
        items[j], items[i] = items[i], items[j]
