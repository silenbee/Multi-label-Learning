import tensorflow as tf 
import numpy as np 
import os
import matplotlib.image as img
import matplotlib.pyplot as plt


sess = tf.InteractiveSession()

# way 1
# image = img.imread('./1.jpg')
# image = tf.cast(image, tf.float32)
# d_image = tf.random_crop(image, [224, 224, 3])

# way 2
image = tf.gfile.FastGFile('./1.jpg', 'rb').read()
image = tf.image.decode_jpeg(image, channels=3)
d_image = tf.image.convert_image_dtype(image, tf.float32)

d_image = tf.random_crop(d_image, [224, 224, 3])
d_image = tf.image.convert_image_dtype(d_image, tf.uint8)
# encoded_image = tf.image.encode_jpeg(image)
image = tf.image.random_flip_left_right(image)
image = tf.image.random_flip_up_down(image)

fig = plt.figure()
fig1 = plt.figure()
 
ax = fig.add_subplot(111)
ax1 = fig1.add_subplot(111)
 
ax.imshow(sess.run(tf.cast(image, tf.uint8)))
ax1.imshow(sess.run(tf.cast(d_image, tf.uint8)))
 
plt.show()
