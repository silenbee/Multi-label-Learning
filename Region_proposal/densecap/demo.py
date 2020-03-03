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
from func import weight_variable, bias_variable, conv2d, gen, RoI, nms
import RPN_model as model

wordtoix = cPickle.load(open("word-index/wordtoix.pkl", "rb"))
ixtoword = cPickle.load(open("word-index/ixtoword.pkl", "rb"))

total_train = []
total_regions = []
total_asfmap = []
#  height and width of the k anchors
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
rpn.load_weights('E:\\Code\\vgg16_weights.npz', 'RPN_model_tf/fasterRcnn.module-800', sess)

img = PIL_Image.open("images/" + "10.jpg")
img_arr = np.array(img.resize([224, 224])).reshape(1, 224, 224, 3)

size = img.size
origin_width = size[0]
origin_height = size[1]
w_scale = width / float(origin_width)
h_scale = height / float(origin_height)


ofs = sess.run(offset, feed_dict={img_input: img_arr})
scr = sess.run(score, feed_dict={img_input: img_arr})

sco = scr.reshape(14, 14, k).transpose(2, 0, 1)
result = ofs.reshape(14, 14, 4 * k).transpose(2, 0, 1)

# store the index of score where score > 0.5, output a shape of (3, num), 3 for the dimension index of matrix score
score_index = np.array((np.where(sco > 0.7)))
print("scoreindex", score_index)

infer = []  # store for the filtered coordinate of bbox, with shape (num after filter, 4)
iscore = []
print("score_index:", score_index)
print("score_index shape:", score_index.shape)
for i in range(score_index.shape[1]):
    bbx_index = i
    bbx_k = score_index[0, bbx_index]
    bbx_y = score_index[1, bbx_index]
    bbx_x = score_index[2, bbx_index]
    Y = (bbx_y * float(600)) / 13   #  average grid point of 800*600 
    X = (bbx_x * float(800)) / 13

    kth = bbx_k
    (h, w) = bbox[kth]
    pos_infer = result[bbx_k * 4:bbx_k * 4 + 4, bbx_y, bbx_x]
    y = Y + pos_infer[0] * h    # x, y original image(800*600 not original) left top coordinate
    x = X + pos_infer[1] * w
    h = h * np.exp(pos_infer[2])    # do offset
    w = w * np.exp(pos_infer[3])
    y = y - h / 2   # center coordinate
    x = x - w / 2
    if x < 0 or y < 0 or h < 5 or w < 5 or x + w > 800 or y + h > 600:  #  over the border
        continue
    infer.append([y, x, h, w])
    iscore.append([sco[bbx_k, bbx_y, bbx_x]])

infer = np.array(infer).reshape(-1, 4)
iscore = np.array(iscore).reshape(1, -1)
num = infer.shape[0]
infer = tf.cast(infer, tf.float32)
iscore = tf.cast(iscore, tf.float32)
nms_infer, nms_score = nms(infer, iscore, num)  #doing nms
nms_infer = sess.run(nms_infer)
nms_score = sess.run(nms_score)

num = nms_infer.shape[0]
nms_infer = tf.cast(nms_infer, tf.float32)

nms_infer = sess.run(nms_infer)

nms_infer = nms_infer.reshape(-1, 4)
train_data = nms_infer  # seems like final candidate box
feature = rpn.conv5_3
fmap = sess.run(feature, feed_dict={img_input: img_arr})

print("fmap:", fmap.shape)
fmap = fmap.reshape(512, 14, 14)
fmap = fmap.transpose(1, 2, 0)

# sfmap pojected from fmap to the feature map (ori->feat)
asfmap = [] # store sfamp to RNN
for i in range(train_data.shape[0]):
    Y, X, H, W = train_data[i, :4]
    # print("Y,X,H,W:",Y,X,H,W)
    w_scale = float(14) / width
    h_scale = float(14) / height
    y = int(round(Y * h_scale))
    x = int(round(X * w_scale))
    h = int(round(H * h_scale + (h_scale - 1)))
    w = int(round(W * w_scale + (w_scale - 1)))
    # print("y,x,h,w:",y,x,h,w)
    sfmap = fmap[y:y + h + 1, x:x + w + 1, :]
    input_y = sfmap.shape[0]
    input_x = sfmap.shape[1]
    # print("sfamp:", sfmap.shape)
    sfmap = sfmap.reshape(1, input_y, input_x, 512)
    sfmap = tf.cast(sfmap, tf.float32)
    sfmap = RoI(sfmap, input_y, input_x)
    sfmap = sess.run(sfmap)
    asfmap.append(sfmap)
asfmap = np.concatenate(asfmap, axis=0)
sess.close()
tf.reset_default_graph()

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.imshow(img)
for i in range(train_data.shape[0]):
    Y, X, H, W = train_data[i, :4]
    if i == 0:
        sc = gen(asfmap[i].reshape(1, 2048), None, ixtoword)
    else:
        sc = gen(asfmap[i].reshape(1, 2048), True, ixtoword)
    ax = plt.gca()
    ax.add_patch(Rectangle((X, Y),
                           W,
                           H,
                           fill=False,
                           edgecolor='red',
                           linewidth=3))
    ax.text(X, Y, sc, style='italic', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
fig = plt.gcf()
plt.tick_params(labelbottom='off', labelleft='off')
plt.show()