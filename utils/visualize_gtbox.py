"""Data format must be the same as PascalVOC."""
import tensorflow as tf
import cv2
import os

flags = tf.app.flags

flags.DEFINE_string('img_dir', 'D:\\University\\Data\\new\\imgs',
                    'Path where the imgs live')
flags.DEFINE_string('ans_dir', 'D:\\University\\Data\\new\\Annotations',
                    'Path where the Annotations live')

FLAGS = flags.FLAGS


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


def visualize_gtbox(img, coordinates):
    """TOO."""
    x_min, y_min, x_max, y_max = coordinates[0] - \
        75, coordinates[1] - 75, coordinates[0] + 75, coordinates[1] + 75

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
    cv2.circle(img, (int(coordinates[0]), int(coordinates[1])), 5, (0, 0, 255), -1)
    cv2.imshow('img', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = input('Please input the file name: ')

    img_name = os.path.join(os.path.expanduser(FLAGS.img_dir), (filename + '.jpg'))
    ans_name = os.path.join(os.path.expanduser(FLAGS.ans_dir), (filename + '.txt'))

    img, coordinates = read_image(img_name, ans_name)
    print('x_center: ', coordinates[0], ' y_center: ', coordinates[1])
    visualize_gtbox(img, coordinates)
