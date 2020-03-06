import tensorflow as tf 
import numpy as np 
import os
import argparse
import pickle
from model import CNN_Encoder, Decoder


shuffle_pool_size = 4000




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

def main(args):
#   trianing 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        data_set_train = get_dataset('my_train.record')
        data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(args.batch_size).repeat()
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle)
        # !!

        imgs = tf.placeholder(tf.float32, [args.batch_size, 224, 224, 3])
        captions = tf.placeholder(tf.float32, [args.batch_size, None])
        cnn_model = CNN_Encoder(imgs)
        imgs_feats = cnn_model.conv5_3
        decoder = Decoder(imgs_feats, captions)
        for epoch in range(args.epoch):
            imgs_batch, captions_batch = sess.run([])
            _, time_step = captions_batch.size()
            embeddings = sess.run([decoder.concat_embedding],feed_dict={})
            for i in range(time_step):
                feas = sess.run([decoder.context], feed_dict = {imgs: imgs_batch,
                                                    captions: captions_batch,
                                                    hx: None})
                inputs = tf.concat([feas, embeddings[:,i,:] ], -1) # -1??
                decoder.hx, decoder.cx = decoder.lstm_cell(inputs) # ???

            loss = None
            sess.run(train_step)
            if epoch % 100 == 0:
                train_loss = sess.run([])


        




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
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')

    