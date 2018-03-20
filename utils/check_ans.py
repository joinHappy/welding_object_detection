"""Data format must be the same as PascalVOC."""
import os
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('ans_dir', 'D:\\University\\GraduationDesign\\Data\\val\\Annotations',
                    'New path to save the imgs and annotatios')

FLAGS = flags.FLAGS


def check_ans():
    """Check annotations file format."""
    os.chdir(FLAGS.ans_dir)  # 进入到 annotations 文件夹

    ans_list = os.listdir()

    for i in ans_list:
        with open(i, 'rt') as file_read:
            line = file_read.read()

        to_list = line.split(' ')
        print(i, ": ", to_list)

        if to_list[-1] == '\n':
            line = line[:-2]

            # with open(i, 'wt') as file_write:
            #    file_write.write(line)
            #    print('Rewrite: ', line)

    return


if __name__ == '__main__':
    check_ans()
