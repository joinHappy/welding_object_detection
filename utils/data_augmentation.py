"""Data format must be the same as PascalVOC."""

import os
<<<<<<< HEAD
import cv2
import random

import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('img_dir', 'D:\\University\\train\\imgs',
                    'Path where the imgs live')
flags.DEFINE_string('ans_dir', 'D:\\University\\train\\Annotations',
                    'Path where the Annotations live')
flags.DEFINE_string('new_ans_dir', '',
                    'Path where the new annotations file save')
flags.DEFINE_string('new_img_dir', '',
=======
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
>>>>>>> parent of dbeaf18... Merge branch 'dev' of https://github.com/Sophistt/welding_object_detection into dev
                    'Path where the new image file save')

FLAGS = flags.FLAGS


<<<<<<< HEAD
def read_image(path_to_img, path_to_ans):
    """Todo."""
    img = cv2.imread(path_to_img)

    with open(path_to_ans, 'rt') as fid:
        line = fid.read()
        to_list = line.split(" ")

    x_center, y_center, types = float(to_list[1]), float(
        to_list[2]), int(to_list[-1])
    # print('img_name: ', path_to_img, 'ans_name: ', path_to_ans, 'gt id: ', types)

    return img, [x_center, y_center, types]


def random_horizontal_flip(img, coordinates):
=======
def random_horizontal_flip(image,
							annotations):
>>>>>>> parent of dbeaf18... Merge branch 'dev' of https://github.com/Sophistt/welding_object_detection into dev
    """TODO."""
    return 0


def random_vertical_flip():
    """TODO."""
    return 0


<<<<<<< HEAD
def random_shift_image(img, coordinates):
    """TODO."""
    x_min, y_min, x_max, y_max = coordinates[0] - \
        75, coordinates[1] - 75, coordinates[0] + 75, coordinates[1] + 75

    rows, cols, channels = img.shape

    shift_matrix = np.float32([[1, 0, 100],
                               [0, 1, 200]])

    shift_img = cv2.warpAffine(img, shift_matrix, (cols, rows))

    cv2.imshow('img', img)
    cv2.imshow('shift_img', shift_img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    return 0


def random_rotate_image(img, coordinates):
    """TODO."""
    x_min, y_min, x_max, y_max = coordinates[0] - \
        75, coordinates[1] - 75, coordinates[0] + 75, coordinates[1] + 75

    rows, cols, channels = img.shape

    rotate_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    rotate_img = cv2.warpAffine(img, rotate_matrix, (cols, rows))

    cv2.imshow('img', img)
    cv2.imshow('rotate_img', rotate_img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    return 0


def add_gauss_noise(img, coordinates, mu, sigma):
    """TODO."""
    x_min, y_min, x_max, y_max = coordinates[0] - \
        75, coordinates[1] - 75, coordinates[0] + 75, coordinates[1] + 75

    rows, cols, channels = img.shape

    noisy_img = np.zeros(img.shape, np.uint8)

    for i in range(rows):
        for j in range(cols):
            noisy_img[i, j, 0] = np.clip((img[i, j, 0] + random.gauss(mu, sigma)), 0, 255)
            noisy_img[i, j, 1] = np.clip((img[i, j, 1] + random.gauss(mu, sigma)), 0, 255)
            noisy_img[i, j, 2] = np.clip((img[i, j, 2] + random.gauss(mu, sigma)), 0, 255)

    cv2.imshow('img', img)
    cv2.imshow('noisy_img', noisy_img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

    return 0


def random_adjust_contrast(img, coordinates):
    """TODO."""
    x_min, y_min, x_max, y_max = coordinates[0] - \
        75, coordinates[1] - 75, coordinates[0] + 75, coordinates[1] + 75

    rows, cols, channels = img.shape

    alpha = 2.0
    beita = 125 * (1.0 - alpha)
    ad_brightness_img = np.uint8(np.clip((alpha * img + beita), 0, 255))

    cv2.imshow('img', img)
    cv2.imshow('ad_brightness_img', ad_brightness_img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
=======
def random_shift_image():
    """TODO."""
    return 0


def random_rotate_image():
    """TODO."""
    return 0


def add_noise_image():
    """TODO."""
>>>>>>> parent of dbeaf18... Merge branch 'dev' of https://github.com/Sophistt/welding_object_detection into dev
    return 0


if __name__ == "__main__":
<<<<<<< HEAD
    filename = input('Please input the file name: ')

    img_name = os.path.join(os.path.expanduser(FLAGS.img_dir), (filename + '.jpg'))
    ans_name = os.path.join(os.path.expanduser(FLAGS.ans_dir), (filename + '.txt'))

    img, coordinates = read_image(img_name, ans_name)
    print('x_center: ', coordinates[0], ' y_center: ', coordinates[1])

    # random_shift_image(img, coordinates)
    # random_rotate_image(img, coordinates)
    # random_adjust_contrast(img, coordinates)
    add_gauss_noise(img, coordinates, 0, 20)
=======

    return 0
>>>>>>> parent of dbeaf18... Merge branch 'dev' of https://github.com/Sophistt/welding_object_detection into dev
