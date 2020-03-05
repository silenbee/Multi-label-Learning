import tensorflow as tf
import numpy as np
import cv2

slim_example_decoder = tf.contrib.slim.tfexample_decoder

def parse_tf(example_proto):
    dics = {}
    #定长数据解析
    dics['image/encoded'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['image/width'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    dics['image/height'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    dics['image/filename'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['image/format'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['image/depth'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    # dics['image/object_number']= tf.FixedLenFeature(shape=[], dtype=tf.int64)
 
    #列表数据解析
    # dics["image/object/names"] = tf.VarLenFeature(tf.string)
    # dics['image/object/id'] = tf.VarLenFeature(tf.int64)
    dics["image/object/class/text"] = tf.VarLenFeature(tf.string)
    dics["image/object/class/label"] = tf.VarLenFeature(tf.int64)
    parse_example = tf.parse_single_example(serialized=example_proto,features=dics)
    # object_number = parse_example["image/object_number"]
    
    image = tf.decode_raw(parse_example['image/encoded'],out_type=tf.uint8)
    w = parse_example['image/width']
    h = parse_example['image/height']
    caption = parse_example['image/object/class/text']
    name = parse_example['image/filename']
    # imageslim = slim_example_decoder.Image(
    #       image_key='image/encoded', format_key='image/format', channels=3)
    c = parse_example['image/depth']
    return image,w,h,c,caption, name  # object_number,xmin,xmax,ymin,ymax
 
dataset = tf.data.TFRecordDataset(r"E:\training_data\cnn-rnn-data-process\my_train1.record")
dataset = dataset.map(parse_tf).batch(1).repeat(1)
 
iterator = dataset.make_one_shot_iterator()
 
next_element = iterator.get_next()
with tf.Session() as session:
    image, w, h, c, caption, name = session.run(fetches=next_element)
    print(caption)
    # print(np.array(imageslim))
    image = np.reshape(image,newshape=[h[0],w[0],c[0]])
    #使用OpenCV绘制表框
    cv2.imshow("s",image)
    cv2.waitKey(0)