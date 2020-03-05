import tensorflow as tf 
import numpy as np 
import os
from tensorflow.nn import rnn_cell

class CNN_Encoder():
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()

        self.feat = self.fc6
    
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
            if i < 29:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------CNN Weight Loaded------------")

class Decoder():
    def __init__(self, feats, captions):
        self.feats = feats
        self.captions = captions
        # self.lengths = lengths

        self.parameters = []
        self.vocab_size = None  # count later
        self.vis_dim = 512
        self.vis_num = 196
        self.hidden_dim = 1024

        self.a_feat = None
        self.a_hiddens = None
        
    def conv(self,name, input_data, out_channel, trainable, bias=True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=False)
            res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            if bias:
                biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=False)
                res = tf.nn.bias_add(res, biases)
            # out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return res


    def attention(self):
        self.att_bias = tf.get_variable("att_bias", [self.vis_num], dtype=tf.float32,trainable=True) #should be zero
        self.att_vw = self.conv("att_vw",self.feats,self.vis_dim,trainable=True,bias=False)
        self.att_hw = self.conv("att_hw",self.a_hiddens,self.vis_dim,trainable=True,bias=False) #unsqeeze
        # self.att_bias = 
        self.att_relu = tf.nn.relu(self.att_vw + self.att_hw + self.att_bias, name="att_full")
        self.att_w = self.conv("att_w", self.att_relu, 1, trainable=True,bias=False)
        self.att_alpha = tf.nn.softmax(self.att_w) # dim=1
        self.context = tf.reduce_sum(tf.matmul(self.a_feat, self.att_alpha), 1)
        

    def decolayer(self):
        self.embeddings = tf.get_variable("embedding", [self.vocab_size, 512])
        self.cap_embed = tf.nn.embedding_lookup(self.embeddings, self.captions)
        self.mean_feats = tf.reduce_mean(self.feats, 1) # unsqueeze
        self.concat_embedding = tf.concat([self.mean_feats, self.cap_embed], 1)

        cell_fun = rnn_cell.BasicLSTMCell
        cell = cell_fun(1024, state_is_tuple=True)
        self.cell = rnn_cell.MultiRNNCell([cell], state_is_tuple=True)
        state = self.cell.zero_state(1024, tf.float32) # !!
        
        output, last_state = tf.nn.dynamic_rnn(cell, self.concat_embedding, initial_state=state, scope="rnn")
        output = tf.nn.dropout(output, 0.5)
        self.softmax_w = tf.get_variable("softmax_w", [1024, self.vocab_size])
        self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        probs = tf.nn.softmax(logits)

    def caption_gen(self):
        batch_size, time_step = self.captions.size() #   may have problems
        self.a_hiddens = tf.zeros(batch_size, 1024)
        for i in range(time_step):
            feas = self.context
            input = tf.concat([feas, self.concat_embedding[:,i,:]], -1) #  may have problem
            # self.a_hiddens,_ = self.lstm(input) # !!
            self.a_hiddens, state = tf.nn.dynamic_rnn(self.cell, input, time_major=False)
            logits = tf.matmul(self.a_hiddens, self.softmax_w) + self.softmax_b
            probs = tf.nn.softmax(logits)