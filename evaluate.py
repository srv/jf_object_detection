
# imports

import scipy.misc
import numpy as np
from scipy import ndimage
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import re
import time as time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# Object detection imports

from utils import label_map_util
from utils import visualization_utils as vis_util


flags = tf.app.flags
flags.DEFINE_string('inference_graph_dir', '', 'Directory where the inference graph is stored.')
flags.DEFINE_string('print_thr', '', 'Printing confidence threshold.')
flags.DEFINE_string('test_dir', '', 'test images folder path.')
flags.DEFINE_string('out_dir', 'results', 'output folder path.')
flags.DEFINE_string('id', '', 'identifier.')
flags.DEFINE_string('n_classes', '', 'number of classes.')
flags.DEFINE_string('labels_file', '', 'path to the class label file')

FLAGS = flags.FLAGS


# --- variables ---


MODEL_NAME = FLAGS.inference_graph_dir
print_thr = float(FLAGS.print_thr)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = FLAGS.labels_file

NUM_CLASSES = int(FLAGS.n_classes)

# --- Load a (frozen) Tensorflow model into memory. ---

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# --- Loading label map ---

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("categories")
print(categories)
print("categoriy_index")
print(category_index)

# --- Helper code ---

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# --- Detection ---

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = FLAGS.test_dir
PATH_TO_OUTPUT_DIR = FLAGS.out_dir
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(1, 6)]

with detection_graph.as_default():

    with tf.Session(graph=detection_graph) as sess:

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  # input and output Tensors
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  # get boxes
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')  # get scores
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')  # get classes
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')  # get number of detections

        times = list()  # to calculate inference times

        for root, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):  # for each folder

            for file in enumerate(files):  # for each file in the folder

                filepath = os.path.join(root, file[1])  # file path

                if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

                    image_np = ndimage.imread(filepath, mode="RGB")  # read image

                    h, w, d = image_np.shape  # get image shape

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    start = time.time()  # start timer

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                    done = time.time()  # stop timer
                    elapsed = done - start  # calculate inference time
                    print('elapsed time: ' + str(elapsed))
                    times.append(elapsed)

                    instances = list()




                    for i in range(int(num)):

                        c = (int(classes[0][i]))
                        c1 = category_index[c]['name']  # get name

                        value = (c1,                            # class
                                 str(scores[0][i]),             # score
                                 str(int((boxes[0][i][1])*w)),  # left
                                 str(int((boxes[0][i][0])*h)),  # top
                                 str(int((boxes[0][i][3])*w)),  # right
                                 str(int((boxes[0][i][2])*h)))  # bottom

                        instances.append(value)

                    identifier = str(FLAGS.id)

                    if not os.path.exists(PATH_TO_OUTPUT_DIR + "/txt" + identifier):
                        os.makedirs(PATH_TO_OUTPUT_DIR + "/txt" + identifier)

                    name_out, ext = os.path.splitext(file[1])

                    file_out = os.path.join(PATH_TO_OUTPUT_DIR + "/txt" + identifier, name_out + ".txt")

                    with open(file_out, 'w') as f:
                        for instance in instances:
                            f.write(instance[0] + " " +
                                    instance[1] + " " +
                                    instance[2] + " " +
                                    instance[3] + " " +
                                    instance[4] + " " +
                                    instance[5] + "\n")

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
                                                                       np.squeeze(scores), category_index, min_score_thresh=print_thr,
                                                                       use_normalized_coordinates=True, line_thickness=2, max_boxes_to_draw=2000)

                    # save image
                    if not os.path.exists(PATH_TO_OUTPUT_DIR + "/output" + identifier):
                        os.makedirs(PATH_TO_OUTPUT_DIR + "/output" + identifier)
                    scipy.misc.imsave(PATH_TO_OUTPUT_DIR + "/output"  + identifier + "/" + name_out + "_o" + ext, image_np)

        # get mean inference time
        times.pop(0)
        m_time = sum(times)/len(times)
        fps = 1/m_time
        print('mean inference time: ' + str(m_time))
        print('fps achieved: ' + str(fps))







