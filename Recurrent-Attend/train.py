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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    print("load vocabulary ...")    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # print(vocab.word2idx['car'])
    print("build data loader ...")
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    print("build the models ...")
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab)+1, args.num_layers).to(device)
    # encoder.load_state_dict(torch.load("models/encoder-0430.ckpt"))
    # decoder.load_state_dict(torch.load("models/0430-decoder-56-50.ckpt")) 


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    learningrate = args.learning_rate
    # min loss
    min_loss = 10000
    min_epoch = 10000
    min_i = 10000

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        loss_epoch = 0.0
        for i, (images, captions, targets, lengths) in enumerate(data_loader):
            # print("caption:", captions)
            # print("targets:", targets)

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            padded_seq = pack_padded_sequence(captions, lengths, batch_first=True)[0]
          
            # Forward, backward and optimize
            features = encoder(images)
            outputs,  predicts = decoder(features, captions)
            padded_pred = pack_padded_sequence(predicts, lengths, batch_first=False)[0]
            
            # print("outputs: ", outputs)
            loss_cls = loss_function(outputs, targets.float().to(device))
            loss = loss_cls + 0.5 * criterion(padded_pred, padded_seq)
            # print("loss: ", loss)
            loss_epoch += loss
            optimizer.zero_grad()
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            print("--------------------------------")

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
   
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'new2-decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                # torch.save(encoder.state_dict(), os.path.join(
                #     args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_epoch = epoch + 1
                min_i = i
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder_min_loss.ckpt'))
                # torch.save(encoder.state_dict(), os.path.join(
                #     args.model_path, 'encoder_min_loss.ckpt')) 

        loss_epoch /= total_step
        print("epoch average loss: {:.4f} ".format(loss_epoch))
        # write log
        with open('./train_log.txt','a+') as f:
            f.write('\n'+str(epoch)+' '+str(loss_epoch.item()))

        if epoch % 20 == 0 and not epoch == 0 :
            learningrate = learningrate * 0.1
            optimizer = torch.optim.Adam(params, lr=learningrate)
    print("min_epoch:{} | min_i: {} | min_loss: {}".format(min_epoch, min_i,min_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/zh_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='E:\\Code\\python\\multi-label\\Multiple-instance-learning-master\\CNN_RNN\\data\\resized2007', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/img_tag.txt', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=50, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=51)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
