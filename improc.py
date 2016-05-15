from __future__ import print_function

import math
import numpy as np

def split(image):
    """Split the image data into the top and bottom half."""
    split_height = image.shape[0] / 2
    return image[:split_height], image[split_height:]

BYTE_MAX = 255
CHANNEL_MAX = np.float32(8)
MAX_RED_VALUE = BYTE_MAX - CHANNEL_MAX
CHANNELS_MAX = CHANNEL_MAX * CHANNEL_MAX
MAX_DEPTH = MAX_RED_VALUE * CHANNELS_MAX

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

def mipmap_imputer(image, strategy=np.mean, scales=None):
    """Fill NaNs with localized aggregate values using mipmaps"""
    # Combination of: http://stackoverflow.com/questions/14549696/mipmap-of-image-in-numpy
    # and: http://stackoverflow.com/questions/5480694/numpy-calculate-averages-with-nans-removed
    scales = scales if scales else [(5,5), (3,2), (2,2), (2,2), (2,2), (2,2), (2,2), (1,2)]
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