"""
Usage:
  # From auxiliary folder (object-detection)
  # Create train data:
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  # Create test data:
  python3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('folder', '', 'Path to the folder')
FLAGS = flags.FLAGS
folder = FLAGS.folder

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'noctiluca':
        return 1
    if row_label == 'pulmo':
        return 1
    if row_label == 'tuberculata':
        return 1
    else:
        print("salgo con: " + str(row_label))


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    #width, height = image.size

    width = int(str.split(str(group[1].width))[1])
    height = int(str.split(str(group[1].height))[1])

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):

    path_data = os.path.join(folder, 'data')
    path_csv = path_data + '/test_labels.csv'

    images_path = "images/test"
    writer = tf.python_io.TFRecordWriter(path_data + '/test.record')
    path = os.path.join(folder, images_path)
    examples = pd.read_csv(path_csv)
    grouped = split(examples, 'filename')

    for group in grouped:

        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()

    path_csv = path_data + '/train_labels.csv'

    images_path = "images/train"
    writer = tf.python_io.TFRecordWriter(path_data + '/train.record')
    path = os.path.join(folder, images_path)
    examples = pd.read_csv(path_csv)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print('Successfully created the TFRecords')


if __name__ == '__main__':
    tf.app.run()

