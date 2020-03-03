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
from func import weight_variable, bias_variable, conv2d, compute_iou, train_neural_network, RoI, nms
import RPN_model as model
wordtoix = cPickle.load(open("word-index/wordtoix.pkl", "rb"))
ixtoword = cPickle.load(open("word-index/ixtoword.pkl", "rb"))

rep_size = 256
len_words = 3000
image_feat_size = 2048
keep = 0.5

input_image_feature = tf.placeholder(tf.float32, [1, image_feat_size])
input_data = tf.placeholder(tf.int64, [1, None])
output_targets = tf.placeholder(tf.int64, [1, None])
keep_prob = tf.placeholder(tf.float32)
feat = tf.placeholder(tf.float32, [])


is_train = 0
total_train = []
total_regions = []
total_asfmap = []
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

img_input = tf.placeholder(tf.float32, [1, 224, 224, 3])
rpn = model.RPN(img_input)
offset = rpn.offset
score = rpn.score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
rpn.load_weights('E:\\Code\\vgg16_weights.npz', 'RPN_model_10/fasterRcnn.module-2400', sess)

description = json.load(open("E:\\training data\\VG2016\\region_descriptions.json", "rb"))
for img_id in range(1, 11):

    img = PIL_Image.open("E:\\training data\\VG2016\\VG_100K\\" + str(img_id) + ".jpg")
    img_arr = np.array(img.resize([224, 224])).reshape(1, 224, 224, 3)

    ofs = sess.run(offset, feed_dict={img_input: img_arr})
    scr = sess.run(score, feed_dict={img_input: img_arr})

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

    sco = scr.reshape(14, 14, k).transpose(2, 0, 1)
    result = ofs.reshape(14, 14, 4 * k).transpose(2, 0, 1)
    score_index = np.array((np.where(sco > 0.5)))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(img)
    infer = []
    iscore = []
    for i in range(score_index.shape[1]):
        bbx_index = i
        bbx_k = score_index[0, bbx_index]
        bbx_y = score_index[1, bbx_index]
        bbx_x = score_index[2, bbx_index]
        Y = (bbx_y * float(600)) / 13
        X = (bbx_x * float(800)) / 13

        kth = bbx_k
        (h, w) = bbox[kth]
        pos_infer = result[bbx_k * 4:bbx_k * 4 + 4, bbx_y, bbx_x]   # offset correspond to score_index
        y = Y + pos_infer[0] * h
        x = X + pos_infer[1] * w
        h = h * np.exp(pos_infer[2])
        w = w * np.exp(pos_infer[3])
        y = y - h / 2
        x = x - w / 2
        if x < 0 or y < 0 or h < 5 or w < 5 or x + w > 800 or y + h > 600:
            continue
        infer.append([y, x, h, w])
        iscore.append([sco[bbx_k, bbx_y, bbx_x]])

    infer = np.array(infer).reshape(-1, 4)
    iscore = np.array(iscore).reshape(1, -1)
    num = infer.shape[0]
    infer = tf.cast(infer, tf.float32)
    iscore = tf.cast(iscore, tf.float32)
    nms_infer, nms_score = nms(infer, iscore, num)
    nms_infer = sess.run(nms_infer)
    nms_score = sess.run(nms_score)

    num = nms_infer.shape[0]
    nms_infer = tf.cast(nms_infer, tf.float32)
    ground_truth_ = tf.cast(ground_truth_, tf.float32)
    nms_iou = compute_iou(ground_truth_, 10, nms_infer, num)
    gt_ = tf.argmax(nms_iou, axis=1)

    #         with tf.device("/gpu:0"):
    nms_infer = sess.run(nms_infer)
    gt_ = sess.run(gt_)
    ground_truth_ = sess.run(ground_truth_)

    nms_infer = nms_infer.reshape(-1, 4)
    gt_ = gt_.reshape(-1, 1)
    train_data = np.concatenate([nms_infer, gt_], axis=1)
    feature = rpn.conv5_3
    fmap = sess.run(feature, feed_dict={img_input: img_arr})
    fmap = fmap.reshape(512, 14, 14)
    fmap = fmap.transpose(1, 2, 0)

    asfmap = []
    for i in range(train_data.shape[0]):
        Y, X, H, W = train_data[i, :4]
        w_scale = float(14) / width
        h_scale = float(14) / height
        y = int(round(Y * h_scale))
        x = int(round(X * w_scale))
        h = int(round(H * h_scale + (h_scale - 1)))
        w = int(round(W * w_scale + (w_scale - 1)))
        sfmap = fmap[y:y + h + 1, x:x + w + 1, :]
        input_y = sfmap.shape[0]
        input_x = sfmap.shape[1]
        sfmap = sfmap.reshape(1, input_y, input_x, 512)
        sfmap = tf.cast(sfmap, tf.float32)
        sfmap = RoI(sfmap, input_y, input_x)
        sfmap = sess.run(sfmap)
        asfmap.append(sfmap)
    asfmap = np.concatenate(asfmap, axis=0)
    total_train.append(train_data)
    total_regions.append(regions)
    total_asfmap.append(asfmap)
sess.close()
print("feature capture done!")
train_neural_network(total_train, total_regions, total_asfmap, wordtoix)