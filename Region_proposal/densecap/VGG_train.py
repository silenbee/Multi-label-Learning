import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 
import tensorflow as tf
import VGG16_model as model
import create_and_read_TFRecord2 as reader2
 
if __name__ == '__main__':
 
    X_train, y_train = reader2.get_file('./train')
    image_batch, label_batch = reader2.get_batch(X_train, y_train, 224, 224, 25, 256)
 
    x_imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_imgs = tf.placeholder(tf.int32, [None, 2])
 
    vgg = model.vgg16(x_imgs)
    fc3_cat_and_dog = vgg.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_cat_and_dog, labels=y_imgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
    correct_prediction = tf.equal(tf.arg_max(y_imgs, 1), tf.arg_max(fc3_cat_and_dog, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    vgg.load_weights('./vgg16_weights.npz', sess)
    saver = vgg.saver()
 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
 
    import time
    start_time = time.time()
 
    for i in range(200):
        image, label = sess.run([image_batch, label_batch])
        labels = reader2.onehot(label)
        sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: labels})
        loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: labels})
        accuracy_record = sess.run(accuracy, feed_dict={x_imgs: image, y_imgs: labels})
        print("the loss is %f " % loss_record, "the accuracy is %f" % accuracy_record)
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
        print("----------epoch %d is finished---------------" % i)
 
    saver.save(sess, "./model/")
    print("Optimization Finished!")
 
 