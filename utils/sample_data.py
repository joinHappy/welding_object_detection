"""Data format must be the same as PascalVOC."""
import os
import random
import shutil
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string('root_dir', 'D:\\University\\GraduationDesign\\Data\\I',
                    'Path where the imgs and Annotations live')
flags.DEFINE_string('new_dir', 'D:\\University\\GraduationDesign\\Data\\test',
                    'New path to save the imgs and annotatios')
flags.DEFINE_integer('sample_number', 300,
                     'Number of images and annotatios being sampled')


FLAGS = flags.FLAGS


def sample_data():
    """todo."""
    ans_dir, img_dir = os.listdir(FLAGS.root_dir)

    ans_dir = os.path.join(os.path.expanduser(FLAGS.root_dir), ans_dir)
    os.chdir(ans_dir)

    all_ans = os.listdir()

    return random.sample(all_ans, FLAGS.sample_number)


def move_file(label_list):
    """todo."""
    img_list = []
    for i in label_list:
        j = i.replace('txt', 'jpg')
        img_list.append(j)

    ans_dir, img_dir = os.listdir(FLAGS.root_dir)

    # move iamges
    for i in img_list:
        file_dir = os.path.join(FLAGS.root_dir, img_dir, i)
        new_dir = os.path.join(FLAGS.new_dir, img_dir, i)
        shutil.move(file_dir, new_dir)
    print('Move all images successfully!')

    # move annotations
    for i in label_list:
        file_dir = os.path.join(FLAGS.root_dir, ans_dir, i)
        new_dir = os.path.join(FLAGS.new_dir, ans_dir, i)
        shutil.move(file_dir, new_dir)
    print('Move all annotations successfully!')


if __name__ == '__main__':
    rand_ans_list = sample_data()
    move_file(rand_ans_list)
