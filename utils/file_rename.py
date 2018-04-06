"""Data format must be the same as PascalVOC."""
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'D:\\有问题数据\\img\\L',
                    'Path where the data save')

FLAGS = flags.FLAGS


def file_rename():
    """Todo."""
    sub_dir = os.listdir(FLAGS.annotations_dir)

    for sub_folder in sub_dir:
        os.chdir(os.path.join(FLAGS.annotations_dir, sub_folder))

        file_list = os.listdir()

        for file in file_list:
            if os.path.isfile(file):
                # new_name = 'L{}{:0>4d}.'.format(int(sub_folder), int(file[:-4]))
                new_name = file.replace('txt', 'jpg')
                os.rename(file, new_name)
                print('Reanme ' + file + ' successfully!')
            else:
                print('Dir is not a file!')


if __name__ == "__main__":
    file_rename()
