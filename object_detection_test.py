"""Run under unix."""
import os
import cv2
import tensorflow as tf
import numpy as np
import time


# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tqdm import trange

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.app.flags

flags.DEFINE_string('model_name', 'frozen_inference_graph.pb',
                    'Saved model name')
flags.DEFINE_string('path_to_model', 'export_model/frozen_inference_graph.pb',
                    'Path to saved model')
flags.DEFINE_string('path_to_labelmap', 'data/welding_label_map.pbtxt',
                    '')
flags.DEFINE_string('path_to_data', 'test',
                    'Path where the img and annotations live')

FLAGS = flags.FLAGS


def classification(img, detection_boxes, detection_scores, detection_classes, detection_nums):
    """TODO."""
    # Expand dimension since the model expects image to have shape [1, None, None, 3].
    img_expanded = np.expand_dims(img, axis=0)

    (boxes, scores, classes, nums) = sess.run(
        [detection_boxes, detection_scores, detection_classes, detection_nums],
        feed_dict={image_tensor: img_expanded})

    return boxes, scores, classes, nums


def load_data():
    """Todo."""
    # img = Image.open(path_to_img)
    sub_dir = os.listdir(FLAGS.path_to_data)

    img_dir = os.path.join(FLAGS.path_to_data, sub_dir[1])
    ans_dir = os.path.join(FLAGS.path_to_data, sub_dir[0])

    img_list = os.listdir(img_dir)
    ans_list = []
    for i in img_list:
        ans_list.append(i.replace('jpg', 'txt'))

    for i in range(len(img_list)):
        img_list[i] = os.path.join(img_dir, img_list[i])
        ans_list[i] = os.path.join(ans_dir, ans_list[i])

    return img_list, ans_list


def read_image(path_to_img, path_to_ans):
    """Todo."""
    img = cv2.imread(path_to_img)

    with open(path_to_ans, 'rt') as fid:
        line = fid.read()
        to_list = line.split(" ")

    x_min, y_min, types = float(to_list[1]), float(to_list[2]), int(to_list[-1])
    # print('img_name: ', path_to_img, 'ans_name: ', path_to_ans, 'gt id: ', types)

    return img, [x_min, y_min, types]


def visualize(img, boxes, classes, scores):
    """TODO."""
    label_map = label_map_util.load_labelmap(FLAGS.path_to_labelmap)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=4, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)

    cv2.imshow('iamge', image)
    cv2.waitKey(2500)


def result_analyze(boxes, classes, scores, gt_box):
    """Todo."""
    max_score_index = np.argmax(np.squeeze(scores))
    

    classes = np.squeeze(classes)
    boxes = np.squeeze(boxes)

    class_id = classes[max_score_index]

    predict_y, predict_x = boxes[max_score_index][0], boxes[max_score_index][1]
    # print('predict_x: ', predict_x * 1280, 'predict_y: ', predict_y * 1024, 'gt_x: ', gt_box[0])

    if gt_box[2] != class_id:
        return -1
    else:
        predict_x = predict_x * 1280
        predict_y = predict_y * 1024

        euclidean_error = np.sqrt(np.square(predict_x - gt_box[0]) + np.square(predict_y - gt_box[1]))

        return euclidean_error


if __name__ == "__main__":

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.path_to_model, 'rb') as fid:
            serialserialized_graph = fid.read()
            od_graph_def.ParseFromString(serialserialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            detection_nums = detection_graph.get_tensor_by_name(
                'num_detections:0')

    image_list = []
    box_list = []
    class_list = []
    score_list = []

    classify_error_counter = 0
    classify_correct_counter = 0 
    total_err = 0
    total_time = 0

    print('Detection begin!')
    img_list, ans_list = load_data()
    for i in trange(len(img_list)):
        img, gt_box = read_image(img_list[i], ans_list[i])

        time_begin = time.time()
        boxes, scores, classes, nums = classification(img, detection_boxes, detection_scores, detection_classes, detection_nums)
        time_end = time.time()

        #print('Detect time: {:.3f}s'.format(time_end - time_begin))
        total_time += (time_end - time_begin)

        error =  result_analyze(boxes, classes, scores, gt_box)
        #print(error)

        if error == -1:
            classify_error_counter += 1
            #visualize(img, boxes, classes, scores)
        else:
            classify_correct_counter +=1
            total_err += error

        #image_list.append(img)
        #box_list.append(boxes)
        #class_list.append(classes)
        #score_list.append(scores)

    print('Mean time to predict: {:.3f}s'.format(total_time / (classify_error_counter + classify_correct_counter)))
    print('Accuracy of classifier: {:.2f}%\n'.format(100 *classify_correct_counter / (classify_correct_counter + classify_error_counter)),
            'Numbers of classify wrongly: {:.0f}'.format(classify_error_counter))
    print('Mean error of bounding box with ground truth box: {:.3f}'.format(total_err / classify_correct_counter))
    # for i in range(len(img_list)):
    #    visualize(img_list[i], box_list[i], class_list[i], score_list[i])
