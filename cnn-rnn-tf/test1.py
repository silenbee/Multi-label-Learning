import tensorflow as tf 

# a = tf.get_variable("hx", [2, 1, 4],initializer=tf.constant_initializer(0.0))

# for i in range(3):
#     print("a shape:", a.shape)
#     b = tf.constant([[4.0, 5.0, 6.0, 7.0],[1.0, 2.0, 3.0, 4.0]])
#     b = tf.expand_dims(b, 1)
#     print("b shape:", b.shape)
#     a = tf.concat([a, b], axis=1)
#     print("a shape:", a.shape)
    

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     c = sess.run(a)
#     print(c)

a = tf.placeholder(tf.float32, shape=[2,1,4], name="a")
b = tf.get_variable("b", [2,1,4], initializer=tf.constant_initializer(0.0))
b = tf.concat([b,a], axis=1)
c = tf.get_variable("c", [2,1,4], initializer=tf.constant_initializer(1.0))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run([b],feed_dict={a:c})
    print(x)