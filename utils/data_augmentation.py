"""Data format must be the same as PascalVOC."""

import os
import cv2
import random

import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('img_dir', 'D:\\University\\Data\\train\\imgs',
                    'Path where the imgs live')
flags.DEFINE_string('ans_dir', 'D:\\University\\Data\\train\\Annotations',
                    'Path where the Annotations live')
flags.DEFINE_string('new_ans_dir', 'D:\\University\\Data\\new\\Annotations',
                    'Path where the new annotations file save')
flags.DEFINE_string('new_img_dir', 'D:\\University\\Data\\new\\imgs',
                    'Path where the new image file save')

FLAGS = flags.FLAGS


def read_image(path_to_img, path_to_ans):
    """Todo."""
    img = cv2.imread(path_to_img)

    with open(path_to_ans, 'rt') as fid:
        line = fid.read()
        to_list = line.split(" ")

    x_center, y_center, types = float(to_list[1]), float(to_list[2]), int(to_list[-1])
    # print('img_name: ', path_to_img, 'ans_name: ', path_to_ans, 'gt id: ', types)

    return img, [x_center, y_center, types]


def generate_img_and_ans(img, x_center, y_center, weld_type, file_name):
    """TODO."""
    img_name = os.path.join(os.path.expanduser(FLAGS.new_img_dir), (file_name + '.jpg'))
    ans_name = os.path.join(os.path.expanduser(FLAGS.new_ans_dir), (file_name + '.txt'))

    if weld_type == 1:
        weld_type_name = 'Lweld'
    elif weld_type == 2:
        weld_type_name = 'Vweld'
    elif weld_type == 3:
        weld_type_name = 'Iweld'
    elif weld_type == 4:
        weld_type_name = 'Oweld'
    else:
        raise RuntimeError('type weld error')

    cv2.imwrite(img_name, img)

    with open(ans_name, 'wt') as fid:
        fid.write(file_name + ' {:.2f} {:.2f} 150 150 '.format(x_center, y_center) + weld_type_name + ' {}'.format(weld_type))


def random_horizontal_flip(img, coordinates):
    """TODO."""
    return 0


def random_vertical_flip():
    """TODO."""
    return 0


def random_shift_image(img, coordinates, img_name):
    """TODO."""
    x_shift = random.randint(50, 200)
    y_shift = random.randint(50, 200)

    rows, cols, channels = img.shape

    shift_matrix = np.float32([[1, 0, x_shift],
                               [0, 1, y_shift]])

    shift_img = cv2.warpAffine(img, shift_matrix, (cols, rows))

    trans_x_center = coordinates[0] + x_shift
    trans_y_center = coordinates[1] + y_shift
    weld_type = coordinates[2]

    new_img_name = img_name + '_x_{}_y_{}'.format(x_shift, y_shift)

    """
    homo_coordinates = np.float16([coordinates[0], coordinates[1], 1])
    trans_matrix = np.float16([[1, 0, 0],
                               [0, 1, 0],
                               [100, 200, 1]])
    trans_coordinates = np.matmul(homo_coordinates, trans_matrix)
    print(trans_coordinates)
    """
    return shift_img, trans_x_center, trans_y_center, weld_type, new_img_name


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
    return 0


if __name__ == "__main__":
    filename = input('Please input the file name: ')

    img_name = os.path.join(os.path.expanduser(FLAGS.img_dir), (filename + '.jpg'))
    ans_name = os.path.join(os.path.expanduser(FLAGS.ans_dir), (filename + '.txt'))

    img, coordinates = read_image(img_name, ans_name)
    print('x_center: ', coordinates[0], ' y_center: ', coordinates[1])

    shift_img, trans_x_center, trans_y_center, weld_type, new_img_name = random_shift_image(img, coordinates, filename)
    generate_img_and_ans(shift_img, trans_x_center, trans_y_center, weld_type, new_img_name)

    # random_rotate_image(img, coordinates)
    # random_adjust_contrast(img, coordinates)
    # add_gauss_noise(img, coordinates, 0, 20)