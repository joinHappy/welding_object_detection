"""Data format must be the same as PascalVOC."""
import os
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'D:\\University\\GraduationDesign\\Data\\O\\Annotations',
                    'Path where the annotations save')

# flags.DEFINE_string('./', '', './')
FLAGS = flags.FLAGS


def ans_fixed():
    """进入到 annotatios 文件夹下轮流读取每个.txt文件，并且修改其中的内容"""
    os.chdir(FLAGS.annotations_dir) 

    ans_list = os.listdir()

    for ans_file in ans_list:
        with open(ans_file, 'rt') as f:
            line = f.read()

        new_line = ans_file[0:-4] + " " + line[:-2] + " Oweld 4"

        with open(ans_file, 'wt') as f_write:
            f_write.write(new_line)
            print("Write " + ans_file)


if __name__ == "__main__":
    ans_fixed()
