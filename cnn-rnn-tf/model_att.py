import tensorflow as tf 
import numpy as np 
import os
from tensorflow.nn import rnn_cell
from tensorflow.contrib.layers.python.layers import initializers

class CNN_Encoder():
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        # self.fc_layers()

        self.feat = self.conv5_3
    
    def maxpool(self,name,input_data, trainable):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out
 
    def conv(self,name, input_data, out_channel, trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=False)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=False)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out
 
    def fc(self,name,input_data,out_channel,trainable = True):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable = trainable)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable = trainable)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases]
        return out
 
    def convlayers(self):
        # zero-mean input
        #conv1
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64,trainable=False)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64,trainable=False)
        self.pool1 = self.maxpool("poolre1",self.conv1_2,trainable=False)
 
        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128,trainable=False)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128,trainable=False)
        self.pool2 = self.maxpool("pool2",self.conv2_2,trainable=False)
 
        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256,trainable=False)
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256,trainable=False)
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256,trainable=False)
        self.pool3 = self.maxpool("poolre3",self.conv3_3,trainable=False)
 
        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512,trainable=False)
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=False)
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=False)
        self.pool4 = self.maxpool("pool4",self.conv4_3,trainable=False)
 
        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512,trainable=False)
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=False)
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5 = self.maxpool("poorwel5",self.conv5_3,trainable=False)
 
    def fc_layers(self):
 
        self.fc6 = self.fc("fc6", self.pool5, 4096,trainable=False)
        self.fc7 = self.fc("fc7", self.fc6, 4096,trainable=False)
        self.fc8 = self.fc("fc8", self.fc7, 2)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i < 26:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------CNN Weight Loaded------------")

class Decoder():
    def __init__(self):
        # self.feats = feats
        # self.captions = tf.cast(captions, dtype=tf.int32)
        self.batch_size = 32
        self.parameters = []
        self.vocab_size = 24  # count later
        self.vis_dim = 512
        self.vis_num = 196
        self.hidden_dim = 1024

        self.initializer = initializers.xavier_initializer()
        # params set
        self.hx = tf.get_variable("hx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        self.cx = tf.get_variable("cx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        
        self.embeddings = tf.get_variable("embedding", [self.vocab_size, 512], initializer=self.initializer)
        self.att_bias = tf.get_variable("att_bias", [self.vis_num], dtype=tf.float32,trainable=True,initializer=tf.constant_initializer(0.0)) #should be zero
        
        self.cell_fun = rnn_cell.BasicLSTMCell
        self.cell_ = self.cell_fun(1024, state_is_tuple=True)
        self.cell = rnn_cell.MultiRNNCell([self.cell_], state_is_tuple=True)
        

    def saver(self):
        # variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['rcnn'])
        # saver = tf.train.Saver(variables_to_restore)
        # return saver
        return tf.train.Saver(self.parameters)

    def fc(self,name,input_data,out_channel,trainable = True, bias=True):
        shape = input_data.get_shape().as_list()
        print("shape:", len(shape))
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        elif len(shape) == 3:
            size = shape[-1] * shape[-2]
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable = trainable,initializer=self.initializer)
            res = tf.matmul(input_data_flat,weights)
            if bias:
                biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable = trainable,initializer=tf.constant_initializer(0.0))
                res = tf.nn.bias_add(res,biases)
                self.parameters += [biases]
            # out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights]
        return res

    def conv(self,name, input_data, out_channel, trainable, bias=True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=False,initializer=self.initializer)
            res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            if bias:
                biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=False,initializer=tf.constant_initializer(0.0))
                res = tf.nn.bias_add(res, biases)
                self.parameters += [biases]
            # out = tf.nn.relu(res, name=name)
        self.parameters += [kernel]
        return res


    # def attention(self, features, hiddens):
    #     att_fea = tf.contrib.layers.fully_connected(inputs=features,num_outputs=self.vis_dim,activation_fn=None,trainable=True,weights_initializer=self.initializer,reuse=tf.AUTO_REUSE,scope="att_v")
    #     att_h = tf.contrib.layers.fully_connected(inputs=hiddens,num_outputs=self.vis_dim,activation_fn=None,trainable=True,weights_initializer=self.initializer,reuse=tf.AUTO_REUSE,scope="att_h")
        
    #     att_full = tf.nn.relu(att_fea + tf.expand_dims(att_h, 1) + tf.reshape(self.att_bias, [1, -1, 1]))
    #     att_out = tf.contrib.layers.fully_connected(inputs=att_full,num_outputs=1,activation_fn=None,trainable=True,weights_initializer=self.initializer,reuse=tf.AUTO_REUSE,scope="att_out")
    #     alpha = tf.nn.softmax(att_out, dim=1)
    #     context = tf.reduce_sum(features*alpha, 1)
    #     return context, alpha

    def forward(self, features, captions):
        cap_embed = tf.nn.embedding_lookup(self.embeddings, captions)
        feats = tf.expand_dims(tf.reduce_mean(features, 1), 1)
        concat_embed = tf.concat([feats, cap_embed], 1)
        
        cx = self.cell.zero_state(self.batch_size, tf.float32)
        output, last_state = tf.nn.dynamic_rnn(self.cell, concat_embed, initial_state=cx)
        print("output.shape: ", output.shape)
        return output
        
     

    def load_weights(self, weight_file, sess):
        self.saver().restore(sess, weight_file)
        print("-----------decoder loaded---------------")


