"""Data format must be the same as PascalVOC."""

import os
import tensorflow as tf
import cv2

flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'D:\\有问题数据\\img\\L',
                    'Path where the annotations file save')
flags.DEFINE_string('imgs_dir', '',
                    'Path where the image file save')
flags.DEFINE_string('new_annotations_dir', '',
                    'Path where the new annotations file save')
flags.DEFINE_string('new_imgs_dir', '',
                    'Path where the new image file save')

FLAGS = flags.FLAGS


def random_horizontal_flip(image,
							annotations):
    """TODO."""
    return 0


def random_vertical_flip():
    """TODO."""
    return 0


def random_shift_image():
    """TODO."""
    return 0


def random_rotate_image():
    """TODO."""
    return 0


def add_noise_image():
    """TODO."""
    return 0


if __name__ == "__main__":

    return 0
