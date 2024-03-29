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
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epoch):
            image, w, h, c, caption,caption_number, name = sess.run(fetches=train_iter)
            image = pre_process(np.array(image).reshape(32, 256, 256, 3))
            image = sess.run(image)
            caption_number = sess.run(tf.sparse_tensor_to_dense(caption_number))

            _, time_step = caption_number.shape # ??
            # decoder.hx = tf.zeros([args.batch_size, 1024])
            # decoder.cx = tf.zeros([args.batch_size, 1024])
            # predicts = tf.zeros([args.batch_size, time_step, args.vocab_size])
            predicts = np.zeros((args.batch_size, time_step, args.vocab_size))  # use numpy instead of tf.zeroes?

            s_img_feats = sess.run([imgs_feats], feed_dict={imgs: image})
            s_img_feats = np.array(s_img_feats).reshape(32, 196, 512)
            embeddings = sess.run([decoder.concat_embedding],feed_dict={cnn_feats: s_img_feats,captions: caption_number})
            embeddings = np.array(embeddings).reshape(args.batch_size, -1, 512)
            print("embedding shape:", embeddings.shape)

            cell_fun = rnn_cell.BasicLSTMCell
            cell = cell_fun(1024, state_is_tuple=True)
            cell = rnn_cell.MultiRNNCell([cell], state_is_tuple=True)
            cx = cell.zero_state(32, tf.float32)
            print(cx)
            for i in range(time_step):
                feas = sess.run([decoder.context], feed_dict = {cnn_feats: s_img_feats,
                                                    captions: caption_number,
                                                    })
                feas = np.array(feas).reshape(args.batch_size, 512)
                inputs = np.concatenate((feas, embeddings[:,i,:]),axis=-1).reshape(32, 1, 1024)
                # inputs = tf.concat([feas, embeddings[:,i,:]], axis=-1) # -1??
                print("inputs shape", tf.convert_to_tensor(inputs).shape)
                print("inputs:", inputs)
                tmp = [rnn_cell.LSTMStateTuple(tf.convert_to_tensor(sess.run([decoder.hx])), tf.convert_to_tensor(sess.run([decoder.cx])))]
                print("tmp: ", tmp)
                decoder.hx, decoder.cx = tf.nn.dynamic_rnn(decoder.cell, inputs, initial_state = tmp, scope='rnnlm') # ??? time major false
                output = sess.run([decoder.linear], feed_dict = {cnn_feats: s_img_feats,
                                                    captions: caption_number,
                                                    })
                print("output: ", np.array(output))
                print("output.shape:", tf.convert_to_tensor(np.array(output)).shape)   # (1, 32, 24)
                predicts[:,i,:] = np.array(output).reshape(32, 24)
            
            # print("predicts:", predicts)
            print("predict shape: ", tf.convert_to_tensor(np.array(predicts)).shape) # (32, 5, 24)

            # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([predicts], [], [tf.ones_like(targets, dtype=tf.float32)], args.vocab_size)
            # cost = tf.reduce_mean(loss)
            # learning_rate = args.learning_rate
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # train_op = optimizer.apply_gradients(zip(grads, tvars))

            # train_loss, _ = sess.run([cost, train_op],
            #                                     feed_dict={})

            # if (epoch + 1) % 20 == 0:
            #     print(epoch, "training loss: ", train_loss)

            # if (epoch + 1) % 50 == 0:   
            #     decoder.saver().save(sess, 'CR_model/test.module')
    print("train end!")

        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help="size for randomly cropping image")
    parser.add_argument('--img_dir', type=str, default='data/resized_img', help='dir of dataset image')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_cap.txt', help='path of caption file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving params')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch num')
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

    