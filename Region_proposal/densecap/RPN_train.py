import caffe
from tensorflow.contrib import seq2seq
import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell
import time
import io
import _pickle as cPickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
from io import StringIO
import json
from func import weight_variable, bias_variable, conv2d, img2feat, compute_iou, generate_anchors,\
    generate_proposals, split_proposals, centerize_ground_truth, get_gt_param, get_offset_labels
import RPN_model as model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

boxes = tf.Variable([
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
], dtype=tf.float32)
bbox = [
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
]

conv_height = 14
conv_width = 14
height = 600
width = 800
k = 12
gt_num = 10
anchors_num = k * conv_height * conv_width
img_num = 11
img_Epoch = 2000

# description = json.load(open("/home/xiaosucheng/Data/VG/region_descriptions.json", "rb"))
description = json.load(open("E:\\training data\\VG2016\\region_descriptions.json", "rb"))

img_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
rpn = model.RPN(img_input)

# every anchor correspond to an offset
anchors = generate_anchors(boxes, height, width, conv_height, conv_width)
anchors = tf.reshape(anchors, [-1, 4])
ground_truth_pre = tf.placeholder(tf.float32, [None, 4])
ground_truth = centerize_ground_truth(ground_truth_pre)
iou = compute_iou(ground_truth, gt_num, anchors, anchors_num)
positive, negative = split_proposals(anchors, iou, rpn.score)
positive_bbox, positive_scores, positive_labels = positive
negative_bbox, negative_scores, negative_labels = negative

predicted_scores = tf.concat([positive_scores, negative_scores], 0)
true_labels = tf.concat([positive_labels, negative_labels], 0)
score_loss = tf.reduce_sum(tf.square(predicted_scores - true_labels))   # if proposal is positive then score should close to 1, negative score close to 0

gt_param = get_gt_param(ground_truth, gt_num, anchors, anchors_num)
pos_offset, pos_offset_labels = get_offset_labels(gt_param, gt_num, rpn.offset, iou)
offset_loss = tf.reduce_sum(tf.square(pos_offset - pos_offset_labels))

total_loss = score_loss + offset_loss
learning_rate = tf.Variable(0.0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
rpn.load_weights('E:\\Code\\vgg16_weights.npz', 'RPN_model_tf/fasterRcnn.module-1800', sess)
saver = rpn.saver()


print("start training ...")
for img_epoch in range(img_Epoch):
    train_img = np.arange(1, img_num, 1)
    np.random.shuffle(train_img)
    sess.run(tf.assign(learning_rate, 0.00001))
    for img_id in train_img:
        img = PIL_Image.open("E:\\training data\\VG2016\\VG_100K\\" + str(img_id) + ".jpg")
        regions = description[img_id - 1]["regions"]
        size = img.size
        origin_width = size[0]
        origin_height = size[1]
        w_scale = width / float(origin_width)
        h_scale = height / float(origin_height)
        ground_truth_ = []
        for idx in range(gt_num):
            rgt = [int(round(regions[idx]["y"] * h_scale)),
                   int(round(regions[idx]["x"] * w_scale)),
                   int(round(regions[idx]["height"] * h_scale + (h_scale - 1))),
                   int(round(regions[idx]["width"] * w_scale + (w_scale - 1)))]
            ground_truth_.append(rgt)
        ground_truth_ = np.array((ground_truth_))
        input = np.array(img.resize([224, 224])).reshape(1, 224, 224, 3)
        Epoch = 3
        with tf.device("/gpu:0"):
            for epoch in range(Epoch):
                sess.run([train_step], feed_dict={img_input: input, ground_truth_pre: ground_truth_})
                #                 if epoch%(Epoch/10) == 0:
    if img_epoch % (img_Epoch / 100) == 0:
        print("epoch:", img_epoch, "img:", img_id, sess.run([total_loss],
                                                            feed_dict={img_input: input,
                                                                       ground_truth_pre: ground_truth_}))
    if (img_epoch + 1) % (img_Epoch / 10) == 0:
        saver.save(sess, 'RPN_model_tf/fasterRcnn.module', global_step=img_epoch + 1)
sess.close()
print("train end")