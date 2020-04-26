import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from PIL import Image
from data_loader_v2 import get_loader 
from build_vocab_v2 import Vocabulary
from model_attention import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from torch.autograd import Variable as V

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = Vocabulary()

with open('./data/zh_vocab_2007.pkl', 'rb') as f:
        vocab = pickle.load(f)

transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
                            ])

encoder_dict = './models/encoder-init-2007.ckpt'
decoder_dict = './models/new-decoder-29-50.ckpt'
img = Image.open('./000288.jpg').convert('RGB')
img = transform(img)
img = torch.unsqueeze(img, dim = 0)
# print(img.shape)
 
encoder = EncoderCNN(512).to(device)#导入网络模型
decoder = DecoderRNN(512, 1024, 24, 1).to(device)
encoder.eval()
decoder.eval()
encoder.load_state_dict(torch.load(encoder_dict))#加载训练好的模型文件
decoder.load_state_dict(torch.load(decoder_dict))#加载训练好的模型文件

# input = V(img.cuda())
features = encoder(img.to(device))
outputs = decoder.sample(features)

sampled_ids = outputs[0][0].data.cpu().numpy().tolist()
sampled_caption = []
for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
                continue
        if word == '<end>':
                break
        sampled_caption.append(word)
        
print(sampled_caption)
