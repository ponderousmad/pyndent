from __future__ import print_function

import errno
import gzip
import math
import os

import numpy as np
import skimage.color

from scipy import ndimage
from six.moves import zip_longest
from six.moves import cPickle as pickle

def split(image):
    """Split the image data into the top and bottom half."""
    split_height = image.shape[0] / 2
    return image[:split_height], image[split_height:]

BYTE_MAX = 255
CHANNEL_MAX = np.float32(8)
MAX_RED_VALUE = BYTE_MAX - CHANNEL_MAX
CHANNELS_MAX = CHANNEL_MAX * CHANNEL_MAX
MAX_DEPTH = MAX_RED_VALUE * CHANNELS_MAX
COLOR_CHANNELS = 3

def byteToUnit(value):
    return ((2.0 * value) / BYTE_MAX) - 1

def decode_depth(image):
    """~14 bits of depth in millimeters is encoded with 8 bits in red and 3 bits in each of green and blue."""
    orientation = [1, 0, 0, 0] # default orientation if not present in image.

    attitude = {
        "quaternion": [1, 0, 0, 0],
        "euler": [0, 0, 0],
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
    leading_nans = 0

    if np.array_equal(image[0, 0], [BYTE_MAX, 0, 0, BYTE_MAX]):
        # Orientation quaternion is present.
        pixel = image[0, 1]
        orientation = attitude["quaternion"]
        for c in range(len(orientation)):
            orientation[c] = byteToUnit(pixel[c])
        leading_nans += 2

    if np.array_equal(image[0, 2], [BYTE_MAX, 0, 0, BYTE_MAX]):
        # Euler angles and 3x3 rotation matrix are present.
        pixel = image[0, 3]
        orientation = attitude["euler"]
        for c in range(len(orientation)):
            orientation[c] = math.pi * byteToUnit(pixel[c])
        matrix = attitude["matrix"]
        for r in range(len(matrix)):
            pixel = image[0, 4 + r]
            row = matrix[r]
            for c in range(len(row)):
                row[c] = byteToUnit(pixel[c])
        leading_nans += 5

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    depth = ((MAX_RED_VALUE - red) * CHANNELS_MAX) + ((green - red) * CHANNEL_MAX) + (blue - red)

    # Zero in the red channel indicates the sensor provided no data.
    depth[np.where(red == 0)] = np.nan
    # Zero out garbage values from encoded attitude
    depth[0, :leading_nans] = np.nan
    return depth, attitude

def encode_normalized_depth(depth):
    """Given a single normalized depth value, encode it into an RGBA pixel."""
    depth_in_mm = int(depth * MAX_DEPTH)
    red_bits = depth_in_mm / int(CHANNELS_MAX)
    lower_bits = depth_in_mm % int(CHANNELS_MAX)
    green_bits = lower_bits / int(CHANNEL_MAX)
    blue_bits = lower_bits % int(CHANNEL_MAX)
    red = int(MAX_RED_VALUE) - red_bits
    return [red, red + green_bits, red + blue_bits, BYTE_MAX]

def encode_normalized_depths(depths):
    """Given a normalized depth image, encode it into an RGBA image."""
    channel_max = np.int32(CHANNEL_MAX)
    channels_max = np.int32(CHANNELS_MAX)
    int_depths = (depths * MAX_DEPTH).astype(np.int32)
    red_bits = int_depths / channels_max
    lower_bits = np.mod(int_depths, channels_max)
    red = np.uint8(MAX_RED_VALUE) - red_bits.astype(np.uint8)
    green_bits = (lower_bits / channel_max).astype(np.uint8)
    blue_bits = np.mod(lower_bits, channel_max).astype(np.uint8)
    alpha_bits = np.ones_like(red) * np.uint8(BYTE_MAX)
    return np.concatenate(
        [red, red + green_bits, red + blue_bits, alpha_bits],
         axis=len(depths.shape) - 1
    )

def load_image(image_path):
    """Load, split and decode an image."""
    combined_image = ndimage.imread(image_path).astype(np.float32)
    color_image, depth_image = split(combined_image)
    color_image = color_image[:, :, 0 : COLOR_CHANNELS] / BYTE_MAX # Discard alpha and normalize
    depths, attitude = decode_depth(depth_image)
    return (color_image, depths, attitude)

def ascending_factors(number):
    """Calculate the prime factors of a number in ascending order."""
    factor = 2
    while number > 1:
        if number % factor == 0:
            yield factor
            number = number / factor
        else:
            factor += 1

def compute_scales(height, width):
    """Compute the prime factors of a the specified image dimensions, padding with 1 if neccesary."""
    height_scales = reversed(list(ascending_factors(height)))
    width_scales = reversed(list(ascending_factors(width)))
    return list(zip_longest(height_scales, width_scales, fillvalue=1))

def mipmap_imputer(image, strategy=np.mean, smooth=False, scales=None):
    """Fill NaNs with localized aggregate values using mipmaps"""
    # Combination of: http://stackoverflow.com/questions/14549696/mipmap-of-image-in-numpy
    # and: http://stackoverflow.com/questions/5480694/numpy-calculate-averages-with-nans-removed

    # If we weren't provided with scale values, compute them.
    scales = scales if scales else compute_scales(image.shape[0], image.shape[1])

    # Calculate the mipmaps by averaging around NaNs.
    mipmaps = []
    mipmap = image
    for y, x in scales:
        mipmap = mipmap.copy()
        size = mipmap.shape
        reshaped = mipmap.reshape(size[0] / y, y, size[1] / x, x)
        masked = np.ma.masked_array(reshaped, np.isnan(reshaped))
        mipmap = strategy(strategy(masked, axis=3), axis=1).filled(np.nan)
        mipmaps.append(mipmap)

    # Progresively fill in holes in each mipmap scale from the next smaller one.
    for index in reversed(range(len(mipmaps))):
        y, x = scales[index]
        if x > 1:
            mipmap = np.repeat(mipmap, x, axis=1).reshape(mipmap.shape[0], mipmap.shape[1] * x)
        if y > 1:
            mipmap = np.repeat(mipmap, y, axis=0).reshape(mipmap.shape[0] * y, mipmap.shape[1])
        target = mipmaps[index - 1] if index > 0 else image.copy()

        nans = np.where(np.isnan(target))
        target[nans] = mipmap[nans]
        if index > 0 and smooth:
            target = ndimage.filters.gaussian_filter(target, max(y, x))
        mipmap = target

    return target

def compute_mean_depth(files):
    """Given a set of image files, compute the mean of all the depth values."""
    # NOTE: The original version of this function computed the mean of the image means.
    # Since the images have different numbers of missing pixels, this skewed the result slightly.
    depth_sum = 0
    depth_count = 0

    for i, path in enumerate(files):
        _, depth, _ = load_image(path)
        depth_sum += np.nansum(depth)
        depth_count += np.sum(np.isfinite(depth))
        if i % 1000 == 0:
            print("Image", i)
    return depth_sum / depth_count

def compute_std_dev(files, mean):
    """Given a set of image files and the mean depth, compute the standard deviation."""
    variance_sum = 0
    depth_count = 0

    for i, path in enumerate(files):
        _, depth, _ = load_image(path)
        diff = depth - mean
        variance_sum += np.nansum(diff * diff)
        depth_count += np.sum(np.isfinite(depth))
        if i % 1000 == 0:
            print("Image", i)
    return math.sqrt(variance_sum / depth_count)

#CIELAB image component scales:
L_MAX = 100
AB_SCALE_MAX = 127

def rgb2lab_normalized(image):
    """Convert an RGB image to a CIELAB image and normalize it."""
    lab_image = skimage.color.rgb2lab(image)
    return (lab_image / [L_MAX / 2, AB_SCALE_MAX, AB_SCALE_MAX]) - [1, 0, 0]

def process_cached(cache_directory, image_path):
    """Process the image file and cache it, or grab cached result."""
    cached = None
    cache_file = os.path.split(image_path)[1] + ".pickle"
    cache_path = os.path.join(cache_directory, cache_file)
    try:
        with gzip.open(cache_path, 'rb') as f:
            cached = pickle.load(f)
    except KeyboardInterrupt:
        raise # Make sure this makes it out.
    except OSError as e:
        if e.errno != errno.ENOENT:
            print("OSError opening cached image:", e.errno, e)
    except IOError as e:
        if e.errno != errno.ENOENT:
            print("IOError opening cached image:", e.errno, e)
    except Exception as e:
        print("Error opening cached image:", e)

    if cached:
        return cached["image"], cached["depth"]

    # Load and process the image.
    rgb_image, depth, _ = load_image(image_path)
    # Convert from rgb to CIE LAB format.
    lab_image = rgb2lab_normalized(rgb_image)

    if not cached:
        try:
            with gzip.open(cache_path, 'wb') as f:
                cache_data = { "image": lab_image, "depth": depth}
                pickle.dump(cache_data, f, pickle.HIGHEST_PROTOCOL)
        except KeyboardInterrupt:
            raise # Make sure this makes it out.
        except Exception as e:
            print("Error caching image:", image_file, "-", e)
    return lab_image, depth

def lerp(a, b, t):
    """Linearly interploate between a and b by t."""
    return (1.0 - t) * a + t * b;

# Based on http://flafla2.github.io/2014/08/09/perlinnoise.html and
# https://en.wikipedia.org/wiki/Perlin_noise

class PerlinNoise(object):
    """Compute 2D Perlin noise."""
    def __init__(self, height, width, entropy=np.random):
        self.height = height
        self.width = width
        self.setup_noise(entropy)

    def setup_noise(self, entropy):
        """Generate random unit length vectors for each grid point."""
        self.noise_angles = entropy.uniform(0, 2 * math.pi, size=(self.height, self.width))
        self.noise_sin = np.sin(self.noise_angles)
        self.noise_cos = np.cos(self.noise_angles)

    def dot_noise(self, iy, ix, y, x):
        """Dot product of noise with local vector."""
        return y * self.noise_sin[iy, ix] + x * self.noise_cos[iy, ix]

    def fade(self, t):
        """Non-linear sigmoid like function for easeing noise shape."""
        return ((6 * t - 15) * t + 10) * np.power(t, 3)

    def at(self, ys, xs):
        """Given 'index' arrays, calculate corresponding noise."""
        ys = np.mod(ys, self.height)
        xs = np.mod(xs, self.width)

        iy0 = np.floor(ys).astype(np.int32)
        ix0 = np.floor(xs).astype(np.int32)

        dy = ys - iy0
        dx = xs - ix0

        iy1 = np.mod(iy0 + 1, self.height)
        ix1 = np.mod(ix0 + 1, self.width)

        sy = self.fade(dy)
        sx = self.fade(dx)

        # Interpolate between grid point gradients
        # Fun fact, the pseudocode for this on wikipedia was wrong, (missing "- 1"s)
        # I tried to fix it, but my change was reverted.
        n00 = self.dot_noise(iy0, ix0, dy,     dx)
        n10 = self.dot_noise(iy1, ix0, dy - 1, dx)
        n01 = self.dot_noise(iy0, ix1, dy,     dx - 1)
        n11 = self.dot_noise(iy1, ix1, dy - 1, dx - 1)
        values = lerp(lerp(n00, n10, sy), lerp(n01, n11, sy), sx)

        return values

    def fill(self, height, width, y_min, y_max, x_min, x_max):
        """Construct a noise image for the given size and range."""
        ys = np.empty(shape=(height, width), dtype=np.float32)
        xs = np.empty_like(ys)
        ys[:, :] = np.linspace(y_min, y_max, num=height, dtype=np.float32)[:,np.newaxis]
        xs[:, :] = np.linspace(x_min, x_max, num=width, dtype=np.float32)[np.newaxis,:]
        return self.at(ys, xs)

def make_noise(height, width, y_scale, x_scale, entropy=np.random):
    """Create a noise image for the given size and scale."""
    noise = PerlinNoise(y_scale, x_scale, entropy)
    return noise.fill(height, width, 0, y_scale, 0, x_scale)

def enumerate_images(root):
    """List all the png files starting from root and split them into test and training sets."""
    training = []
    test = []

    for root, dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(root, name)
            low_name = name.lower()
            # Find all the image files, split into test and training.
            if low_name.endswith(".png"):
                if low_name.endswith("0.png"):
                    test.append(path)
                else:
                    training.append(path)
    return training, test
