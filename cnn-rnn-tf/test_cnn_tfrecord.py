import tensorflow as tf 
import numpy as np 
import os
import argparse

from model import CNN_Encoder, Decoder
from read_record import get_iter



# pre-process image for training
def pre_process(decoded_image):
    # image = tf.gfile.FastGFile('./1.jpg', 'rb').read()
    # decoded_image = tf.image.decode_jpeg(image, channels=3)
    for i in range(32):
        converted_image = tf.image.convert_image_dtype(decoded_image[i], tf.float32)
        converted_image = tf.random_crop(converted_image, [224, 224, 3])
        image = tf.image.random_flip_left_right(converted_image)
        image = tf.image.random_flip_up_down(image)
        std_img = tf.image.per_image_standardization(image)
        if i == 0:
            a = [std_img]
        else:
            a = tf.concat([a, [std_img]], 0)
    return a



def main(args):
#   training 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        # data_set_train = get_dataset('my_train.record')
        # data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(args.batch_size).repeat()
        # data_set_train_iter = data_set_train.make_one_shot_iterator()
        # train_handle = sess.run(data_set_train_iter.string_handle)
        # !!
        train_iter = get_iter()
        image, w, h, c, caption,caption_number, name = sess.run(fetches=train_iter)
        image = pre_process(np.array(image).reshape(32, 256, 256, 3))
        image = sess.run(image)
        print("image shape: ",image.shape)
        caption_number = sess.run(tf.sparse_tensor_to_dense(caption_number))
        print("caption: ", caption_number)
        batch_size, time_step = caption_number.shape
        print("caption size: ", batch_size, time_step)

        # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        # captions = tf.placeholder(tf.float32, [args.batch_size, None])
        # cnn_model = CNN_Encoder(imgs)
        # cnn_model.load_weights('E:\\Code\\vgg16_weights.npz', sess)
        # imgs_feats = cnn_model.conv5_3
        # feat = sess.run([imgs_feats], feed_dict={imgs: image})
        # feat = np.array(feat).reshape(32, 196, 512)
        # print(feat.shape)
        

        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help="size for randomly cropping image")
    parser.add_argument('--img_dir', type=str, default='data/resized_img', help='dir of dataset image')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_cap.txt', help='path of caption file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving params')

    #Moddel parameter
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training ')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    args = parser.parse_args()
    print(args)
    main(args)
    