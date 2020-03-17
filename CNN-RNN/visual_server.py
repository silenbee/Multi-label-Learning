import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab_v2 import Vocabulary
from model_for_visual import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = img_batch.reshape(14, 14, 512)
    print("vis shape", feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    print("row col:", row, col)

    for i in range(0, 25):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(5, 5, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")

def visualize_attention(img_batch):
    feature_map = img_batch.reshape(-1, 14, 14)
    # print("vis shape", feature_map.shape)  # debug
    
    plt.figure()
 
    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic) # only for show all channels 

    for i in range(0, 3):
        feature_one = feature_map[i,:,:]

        # plt.subplot(3, 4, i + 1)
        plt.axis('off')
        plt.imshow(feature_one)
        sub_img_name_with_path = args.image.split('/')[-1].split('.')[0]+'-'+str(i)+'.jpg'
        # path_vec.append(sub_img_name_with_path) # not used
        plt.savefig(sub_img_name_with_path)
 
    # print("save imgs")
    # print(args.image.split('/')[-1].split('.')[0])    # check path


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)

    # decoder stage
    sampled_ids,_,contexts = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        sampled_caption.append(word)
    sentence = ' '.join(sampled_caption)
    
    # visualize attention
    # print("context.shape: ", contexts.shape)  # debug
    visualize_attention(contexts.detach().cpu())

    # Print out the image and the subimg path
    print (sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='E:\\Code\\python\\multi-label\\Multiple-instance-learning-master\\CNN_RNN\\models\\encoder-init-2007.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='E:\\Code\\python\\multi-label\\Multiple-instance-learning-master\\CNN_RNN\\models\\new-decoder-29-50.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='E:\\Code\\python\\multi-label\\Multiple-instance-learning-master\\CNN_RNN\\data\\zh_vocab_2007.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)