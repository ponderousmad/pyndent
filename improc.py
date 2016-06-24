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

def decode_depth(image):
    """~14 bits of depth in millimeters is encoded with 8 bits in red and 3 bits in each of green and blue."""
    orientation = [1, 0, 0, 0] # default orientation if not present in image.
   
    if np.array_equal(image[0, 0], [BYTE_MAX, 0, 0, BYTE_MAX]):
        # Orientation quaternion is present.
        pixel = image[0, 1]
        for c in range(len(orientation)):
            orientation[c] = ((2.0 * pixel[c]) / BYTE_MAX) - 1

        # Clear out the pixels so they don't get interepreted as depth.
        image[0, 0] = [0, 0, 0, BYTE_MAX]
        image[0, 1] = [0, 0, 0, BYTE_MAX]

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    depth = ((MAX_RED_VALUE - red) * CHANNELS_MAX) + ((green - red) * CHANNEL_MAX) + (blue - red)
    
    # Zero in the red channel indicates the sensor provided no data.
    depth[np.where(red == 0)] = float('nan')
    return depth, orientation

def encode_normalized_depth(depth):
    depth_in_mm = int(depth * MAX_DEPTH)
    red_bits = depth_in_mm / int(CHANNELS_MAX)
    lower_bits = depth_in_mm % int(CHANNELS_MAX)
    green_bits = lower_bits / int(CHANNEL_MAX)
    blue_bits = lower_bits % int(CHANNEL_MAX)
    red = int(MAX_RED_VALUE) - red_bits
    return [red, red + green_bits, red + blue_bits, BYTE_MAX]

def encode_normalized_depths(depths):
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
    combined_image = ndimage.imread(image_path).astype(np.float32)
    color_image, depth_image = split(combined_image)
    color_image = color_image[:, :, 0 : COLOR_CHANNELS] / BYTE_MAX # Discard alpha and normalize
    depths, attitude = decode_depth(depth_image)
    return (color_image, depths, attitude)

def ascending_factors(number):
    factor = 2
    while number > 1:
        if number % factor == 0:
            yield factor
            number = number / factor
        else:
            factor += 1

def compute_scales(height, width):
    height_scales = reversed(list(ascending_factors(height)))
    width_scales = reversed(list(ascending_factors(width)))
    return list(zip_longest(height_scales, width_scales, fillvalue=1))

def mipmap_imputer(image, strategy=np.mean, scales=None):
    """Fill NaNs with localized aggregate values using mipmaps"""
    # Combination of: http://stackoverflow.com/questions/14549696/mipmap-of-image-in-numpy
    # and: http://stackoverflow.com/questions/5480694/numpy-calculate-averages-with-nans-removed
    scales = scales if scales else compute_scales(image.shape[0], image.shape[1])
    mipmaps = []
    mipmap = image
    for y, x in scales:
        mipmap = mipmap.copy()
        size = mipmap.shape
        reshaped = mipmap.reshape(size[0] / y, y, size[1] / x, x)
        masked = np.ma.masked_array(reshaped, np.isnan(reshaped))
        mipmap = strategy(strategy(masked, axis=3), axis=1).filled(np.nan)
        mipmaps.append(mipmap)
    
    for index, mipmap in reversed(list(enumerate(mipmaps))):
        y, x = scales[index]
        expanded = mipmap
        if x > 1:
            expanded = np.repeat(expanded, x, axis=1).reshape(expanded.shape[0], expanded.shape[1] * x)
        if y > 1:
            expanded = np.repeat(expanded, y, axis=0).reshape(expanded.shape[0] * y, expanded.shape[1])
        target = mipmaps[index - 1] if index > 0 else image.copy()

        nans = np.where(np.isnan(target))
        target[nans] = expanded[nans]
    return target

def compute_mean_depth(files):
    depth_averages = []

    for path in files:
        _, depth, _ = improc.load_image(path)
        depth_averages.append(np.nanmean(depth))
        if len(depth_averages) % 1000 == 0:
            print("Image", len(depth_averages))
    return np.nanmean(depth_averages)

#CIELAB image component scales:
L_MAX = 100
AB_SCALE_MAX = 127

def rgb2lab_normalized(image):
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
    return (1.0 - t) * a + t * b;

class PerlinNoise(object):
    def __init__(self, height, width, entropy):
        self.height = height
        self.width = width
        self.setup_noise(entropy)

    def setup_noise(self, entropy):
        """Generate random unit length vectors for each grid point."""
        self.noise_grid = np.zeros(shape=(self.height, self.width, 2))
        for y in range(self.height):
            for x in range(self.width):
                angle = entropy.random() * 2 * math.pi
                self.noise_grid[y, x, 0] = math.sin(angle)
                self.noise_grid[y, x, 1] = math.cos(angle)

    def dot_noise(self, iy, ix, y, x):
        return (y * self.noise_grid[iy, ix, 0] + x * self.noise_grid[iy, ix, 1] + 1) / 2

    def fade(self, t):
        return 6 * math.pow(t, 5) - 15 * math.pow(t, 4) + 10 * math.pow(t, 3)

    def at(self, y, x):
        y = y % self.height
        x = x % self.width

        iy0 = int(math.floor(y))
        iy1 = (iy0 + 1) % self.height
        ix0 = int(math.floor(x))
        ix1 = (ix0 + 1) % self.width

        dy = y - iy0
        dx = x - ix0

        sy = self.fade(dy)
        sx = self.fade(dx)

        # Interpolate between grid point gradients
        n00 = self.dot_noise(iy0, ix0, dy - 0, dx - 0)
        n10 = self.dot_noise(iy1, ix0, dy - 1, dx - 0)
        n01 = self.dot_noise(iy0, ix1, dy - 0, dx - 1)
        n11 = self.dot_noise(iy1, ix1, dy - 1, dx - 1)
        value = lerp(lerp(n00, n10, sy), lerp(n01, n11, sy), sx)

        return value

    def fill(self, array, y_min, y_max, x_min, x_max):
        height = array.shape[0]
        width = array.shape[1]
        y_step = (y_max - y_min) / float(height)
        x_step = (x_max - x_min) / float(width)
        for i in xrange(height):
            for j in xrange(width):
                array[i,j] = self.at(y_min + i * y_step, x_min + j * x_step)

def make_noise(image, y_scale, x_scale, entropy):
    noise = PerlinNoise(y_scale, x_scale, entropy)
    noise.fill(image, 0, y_scale, 0, x_scale)
    return image