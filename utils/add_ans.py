"""Data format must be the same as PascalVOC."""
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('additional_ans_dir', 'D:\\University\\Data\\train\\addition_ans\\WCY_VType_Right',
                    'Path where the additional annotations save')
flags.DEFINE_string('original_ans_dir', 'D:\\University\\Data\\train\\Annotations',
                    'Path where the original annotations save')

FLAGS = flags.FLAGS


def read_add_ans(filename):
    """Read the addtional annotations and return the content in it."""
    with open(filename, 'rt') as fid:
        line = fid.read()
        to_list = line.split(' ')

    return to_list


def write_origin_ans(filename, content):
    """Write the content of addtional annotations to original annotations."""
    with open(filename, 'rt') as fid:
        line = fid.readlines()
        origin_to_list = line[-1].split(' ')

    new_line = ' \n' + origin_to_list[0] + ' ' + content[0] + ' ' + content[1] + ' ' + content[2] + ' ' + content[3] + ' ' + origin_to_list[5] + ' ' + origin_to_list[6] + ' \n'
    print('write ' + filename)

    with open(filename, 'at') as fid:
        fid.write(new_line)


if __name__ == '__main__':
    # filename = input('Please input the file name: ')

    list_dir = os.listdir('D:\\University\\Data\\train\\addition_ans\\WCY_VType_Right')

    for filename in list_dir:
        # add_ans_name = os.path.join(os.path.expanduser(FLAGS.additional_ans_dir), (filename + '.txt'))
        add_ans_name = os.path.join(os.path.expanduser(FLAGS.additional_ans_dir), filename)
        ori_ans_name = os.path.join(os.path.expanduser(FLAGS.original_ans_dir), filename)

        content = read_add_ans(add_ans_name)
        write_origin_ans(ori_ans_name, content)
