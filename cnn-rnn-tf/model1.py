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
        self.fc_layers()

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
    def __init__(self, feats, captions):
        self.feats = feats
        self.captions = tf.cast(captions, dtype=tf.int32)
        # self.lengths = lengths
        self.batch_size = 32
        self.parameters = []
        self.vocab_size = 24  # count later
        self.vis_dim = 512
        self.vis_num = 196
        self.hidden_dim = 1024

        self.initializer = initializers.xavier_initializer()

        self.hx_p = tf.placeholder(tf.float32, shape=[self.batch_size, 1024], name="initial_state_h")
        self.hx = tf.get_variable("hx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        self.cx = tf.get_variable("cx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        print("self.cx", self.cx)

        self.attention()
        self.decolayer()
        

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


    def attention(self):
        # self att biaas move to init
        self.att_bias = tf.get_variable("att_bias", [self.vis_num], dtype=tf.float32,trainable=True,initializer=tf.constant_initializer(0.0)) #should be zero
        self.parameters += [self.att_bias]
        with tf.variable_scope('fc'):
            self.att_vw = tf.contrib.layers.fully_connected(inputs=self.feats,num_outputs=self.vis_dim,activation_fn=None,trainable=True)
            self.att_hw = tf.contrib.layers.fully_connected(inputs=self.hx_p,num_outputs=self.vis_dim,activation_fn=None,trainable=True)
        # self.att_vw = self.fc("att_vw",self.feats,self.vis_dim,trainable=True,bias=False)
        # self.att_hw = self.fc("att_hw",self.hx,self.vis_dim,trainable=True,bias=False) #unsqeeze
        print("vw.shape: ",self.att_vw.shape)   # (32, 196, 512)
        print("hw.shape: ",tf.expand_dims(self.att_hw, 1).shape)  # (32, 1, 512)
        print("bias.shape: ",tf.reshape(self.att_bias, [1, -1, 1]).shape) # (1, 196, 1)
        self.att_relu = tf.nn.relu(self.att_vw + tf.expand_dims(self.att_hw, 1) + tf.reshape(self.att_bias, [1, -1, 1]), name="att_full")
        self.att_w = tf.contrib.layers.fully_connected(inputs=self.att_relu,num_outputs=1,activation_fn=None,trainable=True)
        # self.att_w = self.fc("att_w", self.att_relu, 1, trainable=True,bias=False)
        self.att_alpha = tf.nn.softmax(self.att_w, dim=1) # dim=1
        self.context = tf.reduce_sum(self.feats*self.att_alpha, 1) # not matmul , * !!
        

    def decolayer(self):
        self.embeddings = tf.get_variable("embedding", [self.vocab_size, 512], initializer=self.initializer)
        self.parameters += [self.embeddings]
        self.cap_embed = tf.nn.embedding_lookup(self.embeddings, self.captions)
        self.mean_feats = tf.expand_dims(tf.reduce_mean(self.feats, 1), 1) # unsqueeze
        self.concat_embedding = tf.concat([self.mean_feats, self.cap_embed], 1)

        # self.hx = tf.get_variable("hx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        # self.cx = tf.get_variable("cx", [self.batch_size, 1024],initializer=tf.constant_initializer(0.0))
        self.linear = tf.contrib.layers.fully_connected(inputs=self.hx_p,num_outputs=self.vocab_size,activation_fn=None,trainable=True)
        # self.linear = self.fc("linear",self.hx,self.vocab_size,trainable=True,bias=True)

        self.cell_fun = rnn_cell.BasicLSTMCell
        self.cell_ = self.cell_fun(1024, state_is_tuple=True)
        self.cell = rnn_cell.MultiRNNCell([self.cell_], state_is_tuple=True)
        self.parameters += [self.cell.weights]
        self.init_cell_state = self.cell.zero_state(self.batch_size, tf.float32) # !!
        # print("init_cell shape:", self.init_cell_state.shape)
        # output, last_state = tf.nn.dynamic_rnn(cell, self.concat_embedding, initial_state=state, scope="rnn")
        # output = tf.nn.dropout(output, 0.5)
        # self.softmax_w = tf.get_variable("softmax_w", [1024, self.vocab_size])
        # self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        # logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        # probs = tf.nn.softmax(logits)

    def load_weights(self, weight_file, sess):
        self.saver().restore(sess, weight_file)
        print("-----------decoder loaded---------------")


    # def caption_gen(self):
    #     batch_size, time_step = self.captions.size() #   may have problems
    #     self.a_hiddens = tf.zeros(batch_size, 1024)
    #     for i in range(time_step):
    #         feas = self.context
    #         input = tf.concat([feas, self.concat_embedding[:,i,:]], -1) #  may have problem
    #         # self.a_hiddens,_ = self.lstm(input) # !!
    #         self.a_hiddens, state = tf.nn.dynamic_rnn(self.cell, input, time_major=False)
    #         logits = tf.matmul(self.a_hiddens, self.softmax_w) + self.softmax_b
    #         probs = tf.nn.softmax(logits)