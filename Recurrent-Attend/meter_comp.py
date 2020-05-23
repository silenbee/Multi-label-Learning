import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_v2 import get_loader 
from build_vocab_v2 import Vocabulary
from model_attention import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import loss_function
from utils import AveragePrecisionMeter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    print("load vocabulary ...")    
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # print(vocab.word2idx['car'])
    print("build data loader ...")

    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    print("build the models ...")

    encoder_dict = 'models/loss-with-sigmoid/encoder-6.ckpt'
    decoder_dict = 'models/loss-with-sigmoid/decoder-lstm-with-sigmoid-0512-65-50.ckpt'
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab)+1, args.num_layers).to(device)
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_dict))#加载训练好的模型文件
    decoder.load_state_dict(torch.load(decoder_dict))#加载训练好的模型文件

    ap_meter = AveragePrecisionMeter()

    # compute model precision
    with torch.no_grad():
        for i, (images, captions, targets, lengths) in enumerate(data_loader):
            # print("caption:", captions)
            # print("targets:", targets)

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs,  predicts = decoder.test_batch(features)

            # print("outputs: ", outputs)
            # print("predicts: ", predicts)
            prob = torch.sigmoid(outputs)
            # print("prob: ", prob)
            print("---------------{}----------------".format(i))
            ap_meter.add(prob, targets)
            # map = 100 * ap_meter.value().mean()
            # print("map: ", map)
        
        map = 100 * ap_meter.value()[1:].mean()
        OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
        print('lstm with sigmoid > 0.8 loss version \n')
        print('ap: ', ap_meter.value())
        print('mAP {map:.3f}'.format(map=map))
        print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/zh_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='E:\\training_data\\resized2007_test', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='E:\\training_data\\voc_2007_test.txt', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=50, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=51)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
