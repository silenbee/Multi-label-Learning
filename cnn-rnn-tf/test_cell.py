import tensorflow as tf 
import numpy as np 
import os
import argparse
import pickle
from model import CNN_Encoder, Decoder
from read_record import get_iter
from tensorflow.nn import rnn_cell

shuffle_pool_size = 4000


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
            imgs = [std_img]
        else:
            imgs = tf.concat([imgs, [std_img]], 0)
    return imgs


# eval
def evaluate():
    pass

def main(args):
#   trianing 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        # data_set_train = get_dataset('my_train.record')
        # data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(args.batch_size).repeat()
        # data_set_train_iter = data_set_train.make_one_shot_iterator()
        # train_handle = sess.run(data_set_train_iter.string_handle)
        # # !!
        train_iter = get_iter()
        imgs = tf.placeholder(tf.float32, [args.batch_size, 224, 224, 3])
        captions = tf.placeholder(tf.float32, [args.batch_size, None])
        cnn_feats = tf.placeholder(tf.float32, [args.batch_size, 196, 512])
        cnn_model = CNN_Encoder(imgs)
        cnn_model.load_weights('E:\\Code\\vgg16_weights.npz', sess)
        imgs_feats = cnn_model.conv5_3

        decoder = Decoder(cnn_feats, captions)

        # loop def          for i in range(captions.shape[1]):
        j = 0
        feas = decoder.context
        input = tf.concat((feas, decoder.concat_embedding[:,j,:]), axis=-1)
        input = tf.reshape(input, [32,1,1024])
        print("input.shape: ", input.shape)
        hx, cx = tf.nn.dynamic_rnn(decoder.cell, input, initial_state=decoder.init_cell_state)
        op = tf.assign(decoder.hx, tf.reshape(hx,[32, 1024]))
        output = decoder.linear


        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epoch):
            image, w, h, c, caption,caption_number, name = sess.run(fetches=train_iter)
            image = pre_process(np.array(image).reshape(32, 256, 256, 3))
            image = sess.run(image)
            caption_number = sess.run(tf.sparse_tensor_to_dense(caption_number))

            s_img_feats = sess.run([imgs_feats], feed_dict={imgs: image})
            s_img_feats = np.array(s_img_feats).reshape(32, 196, 512)
            
            predicts = np.zeros((args.batch_size, caption_number.shape[1], args.vocab_size))

            j = 0
            for i in range(caption_number.shape[1]):
                sess.run([op], feed_dict={cnn_feats: s_img_feats, captions: caption_number})
                out = sess.run([output], feed_dict={cnn_feats: s_img_feats, captions: caption_number})
                predicts[:,i,:] = np.array(out).reshape(32, 24)
            
            print(predicts)
            targets = np.array(caption_number).reshape(-1)
            predicts = np.array(predicts).reshape(-1, 24)
            print("predict shape: ", tf.convert_to_tensor(np.array(predicts)).shape)
            print("target shape: ", tf.convert_to_tensor(np.array(targets)).shape)
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([predicts], [targets], [tf.ones_like(targets, dtype=tf.float64)], args.vocab_size)
            cost = tf.reduce_mean(loss)
            learning_rate = args.learning_rate
            tvars = decoder.trainable_var()
            # tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars))

            train_loss, _ = sess.run([cost, train_op],
                                                feed_dict={})
            print("train_loss: ", train_loss)

            print("done once")

                

           
                
            

    print("train end!")

        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help="size for randomly cropping image")
    parser.add_argument('--img_dir', type=str, default='data/resized_img', help='dir of dataset image')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_cap.txt', help='path of caption file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving params')
    parser.add_argument('--epoch', type=int, default=3, help='training epoch num')
    #Moddel parameter
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training ')
    parser.add_argument('--vocab_size', type=int, default=24, help='size of vocab')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    # parser.add_argument('--', type=, default=, help='')
    args = parser.parse_args()
    print(args)
    main(args)

    