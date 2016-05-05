import random

class Mutagen(object):
    """Keep track of probabilities and randomness for mutating conv nets."""
    def __init__(self, seed):
        self.toggle_relu = 0.01
        self.change_dropout_rate = 0.05
        self.DROPOUT_GRANULARITY = 4
        self.output_size_factors = [
            (0.02, 0.5),
            (0.01, 0.75),
            (0.01, 0.9),
            (0.02, 1.1),
            (0.02, 1.25),
            (0.02, 2.0)
        ]
        self.change_distribution = 0.01
        self.initial_means = [
            (0.03, 0),
            (0.02, 1),
            (0.01, -1)
        ]
        self.image_operations = [
            (0.05, "conv_bias"),
            (0.01, "conv"),
            (0.03, "max_pool"),
            (0.03, "avg_pool")
        ]
        self.patches = [
            (0.005, 1),
            (0.008, 2),
            (0.009, 3),
            (0.010, 4),
            (0.015, 5),
            (0.010, 6),
            (0.009, 7),
            (0.008, 8),
            (0.007, 9),
            (0.006, 10),
            (0.005, 11),
            (0.004, 12),
            (0.003, 13),
            (0.002, 14),
            (0.001, 15)
        ]
        self.strides = [
            (0.005, 1),
            (0.010, 2),
            (0.002, 3),
            (0.001, 4),
            (0.001, 5)
        ]
        self.paddings = [
            (0.05, "SAME"),
            (0.05, "VALID")
        ]
        self.add_image_layer = 0.005
        self.remove_image_layer = 0.004
        self.add_hidden_layer = 0.003
        self.remove_hidden_layer = 0.002
        self.entropy = random.Random(seed)

    def branch(self, bias):
        return self.entropy.random() < bias

    def select(self, choices, default=None):
        value = self.entropy.random()
        threshold = 0
        for entry in choices:
            threshold += entry[0]
            if value < threshold:
                return entry[1]
        return default

    def mutate_relu(self, relu):
        return (not relu) if self.branch(self.toggle_relu) else relu

    def mutate_dropout(self, rate):
        if self.branch(self.change_dropout_rate):
            return (1.0 / self.DROPOUT_GRANULARITY) * self.entropy.randint(0, self.DROPOUT_GRANULARITY - 1)
        return rate

    def mutate_output_size(self, output_size):
        return int(output_size * self.select(self.output_size_factors, 1))

    def mutate_distribution(self, distribution):
        if self.branch(self.change_distribution):
            return self.entropy.choice(DISTRIBUTIONS)
        return distribution

    def mutate_initial_mean(self, mean):
        return self.select(self.initial_means, mean)

    def mutate_initial_scale(self, scale):
        return scale * self.select(self.output_size_factors, 1)

    def mutate_image_operation(self, operation):
        return self.select(self.image_operations, operation)

    def mutate_patch_size(self, patch_size):
        return self.select(self.patches, patch_size)

    def mutate_stride(self, stride):
        return self.select(self.strides, stride)

    def mutate_padding(self, padding):
        return self.select(self.paddings, padding)

    def mutate_duplicate_layer(self, is_image, layer_count):
        probability = self.add_image_layer if is_image else self.add_hidden_layer
        if layer_count > 0 and self.branch(probability):
            return self.entropy.randint(0, layer_count - 1)
        return None

    def mutate_remove_layer(self, is_image, layer_count):
        probability = self.add_image_layer if is_image else self.add_hidden_layer
        if layer_count > 1 and self.branch(probability):
            return self.entropy.randint(0, layer_count - 1)
        return None
