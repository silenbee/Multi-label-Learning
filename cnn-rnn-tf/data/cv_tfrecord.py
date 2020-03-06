#creat_my_data_tf_record.py
import hashlib
import io
import logging
import os
from build_vocab_v2 import Vocabulary
from collections import Counter
from lxml import etree
import PIL.Image
import tensorflow as tf
import pickle

#import dataset_util
#import label_map_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# python cv_tfrecord.py --data_dir=E:\\training_data\\VOCdevkit\\  --set=train
# E:\training data\cnn-rnn-data-process>python cv_tfrecord.py --data_dir=E:\\training data\\VOCdevkit\\  --set=train --output_path= E:\\training data\\cnn-rnn-data-process\\my_train.record  --label_map_path E:\\training data\\VOCdevkit\\label_map.pbtxt
flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', 'E:\\training_data\\cnn-rnn-data-process\\my_train1.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'E:\\training_data\\VOCdevkit\\pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS
 
SETS = ['train', 'val', 'trainval', 'test']

gcount = Counter()

def takefreq(elem):
    global gcount
    return gcount[elem]

def get_counter():
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    with open('./img_tag_2007.txt','r') as file:
        for line in file:
            # id,tokens=json.loads(line)
            tokens = line.split()[1].split(',')
            counter.update(tokens)
    # for word, cnt in counter.items():
    #     print(word,':',cnt)
    global gcount
    gcount = counter


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
        dataset_directory: Path to root directory holding PASCAL dataset
        label_map_dict: A map from string label names to integers ids.
        ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
        image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.
    Returns:
        example: The converted tf.Example.
    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    with open("E:\\training_data\\cnn-rnn-data-process\\zh_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join("E:\\Code\\python\\multi-label\\Multiple-instance-learning-master\\CNN_RNN\\data\\resized2007", data['filename'])
    # full_path = os.path.join(dataset_directory, img_path)
    image = PIL.Image.open(full_path)
    image_bytes = image.tobytes()
    # with tf.gfile.GFile(full_path, 'rb') as fid:
    #     encoded_jpg = fid.read()
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = PIL.Image.open(encoded_jpg_io)
    # if image.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')
    # key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    depth = int(data['size']['depth'])
    classes = []
    classes_text = []
    classes_tmp = []
    classes_set = set()
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
    
        difficult_obj.append(int(difficult))
        # classes_text.append(obj['name'].encode('utf8'))
        classes_set.add(obj['name'])
        # classes.append(label_map_dict[obj['name']])
    
    # relist classes and classes_text  
    classes_tmp = list(classes_set)
    classes_tmp.sort(key=takefreq,reverse=True)
    # add <start> 
    classes_text.append('<start>'.encode('utf-8'))
    classes.append(1)
    # append label id
    for i in range(len(classes_tmp)):
        # print(classes_tmp[i])
        classes.append(vocab(classes_tmp[i]))
        classes_text.append(classes_tmp[i].encode('utf-8'))
    # add <end>
    classes_text.append('<end>'.encode('utf-8'))
    classes.append(2)
    # print(classes_text)
    # print(classes)
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/depth': dataset_util.int64_feature(depth),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        # 'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded' : dataset_util.bytes_feature(image_bytes),
        # 'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
    }))
    return example
 
 
def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    
    data_dir = FLAGS.data_dir
    datasets = ['VOC2007']
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    for dataset in datasets:
        logging.info('Reading from PASCAL %s dataset.', dataset)
        examples_path = os.path.join(data_dir, dataset, 'ImageSets', 'Main\\' + FLAGS.set + '.txt')
        print(examples_path)
        annotations_dir = os.path.join(data_dir, dataset, FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        
            tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                            FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())
    
    writer.close()
    print("data record writed")
 
 
if __name__ == '__main__':
    get_counter()
    tf.app.run()
 