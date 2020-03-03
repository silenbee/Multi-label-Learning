import tensorflow as tf 
import numpy as np 
import os
import argparser
import cPickle

# get data from dataset


# pre-process image for training
def pre_porcess(img):
    image = tf.gfile.FastGFile('./1.jpg', 'rb').read()
    decoded_image = tf.image.decode_jpeg(image, channels=3)
    converted_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    image = tf.image.random_flip_left_right(converted_image)
    image = tf.image.random_flip_up_down(image)
    std_img = tf.image.per_image_standardization(img)
    return std_img


# eval
def evaluate():
    pass


#   trianing 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:


if __name__ == '__main__':
    parser = argparser.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help="size for randomly cropping image")
    parser.add_argument('--img_dir', type=str, default='data/resized_img', help='dir of dataset image')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_cap.txt', help='path of caption file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving params')

    #Moddel parameter
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training ')
    parser.add_argument('--', type=, default=, help='')
    parser.add_argument('--', type=, default=, help='')
    parser.add_argument('--', type=, default=, help='')
    parser.add_argument('--', type=, default=, help='')

    