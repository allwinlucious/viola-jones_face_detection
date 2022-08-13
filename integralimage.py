import numpy as np


def integral_image(image):
    """
    In an integral image each pixel is the sum of all pixels in the original image
    that are 'left and above' the pixel.
    :param image:
    :return: integral image of same shape
    """

    height, width = image.shape
    ii = np.pad(image, ((1, 0), (1, 0)), 'constant')
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            ii[y, x] += ii[y - 1, x] + ii[y, x - 1] - ii[y - 1, x - 1]
    return ii[1:, 1:]


def sum_rect_feature(ii, y, x, height, width):
    """
    calculates rectangular feature values
    :param ii: integral image
    :param y: top left coordinate y
    :param x: top left coordinate x
    :param height: height of rectangular region
    :param width: width of rectangular region
    :return: sum of original pixel values in rectangular region
    """
    top_left = (y, x)
    bottom_right = (y + height, x + width)
    top_right = (y, x + width)
    bottom_left = (y + height, x)

    if top_left == bottom_right:
        return ii[top_left]
    return ii[bottom_right] - ii[top_right] - ii[bottom_left] + ii[top_left]
