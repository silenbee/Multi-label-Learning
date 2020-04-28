import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab_v2 import Vocabulary
from pycocotools.coco import COCO

def get_ids_and_caption(file):
    """Build a simple vocabulary wrapper."""
    with open('data/annotations/img_tag.txt','r') as file:
        tokenlist = []
        ids = []
        for line in file:
            # id,tokens=json.loads(line)
            ids.append(line.split()[0])
            tokens = line.split()[1].split(',')
            tokenlist.append(tokens)
        return ids,tokenlist

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        # self.coco = COCO(json)
        self.ids, self.captions = get_ids_and_caption(json)
        #self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.ext_eye = np.vstack((np.zeros(20), np.eye(20)))    # tackle index zero as padding 

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        # caption = coco.anns[ann_id]['caption']
        # img_id = coco.anns[ann_id]['image_id']
        # path = coco.loadImgs(img_id)[0]['file_name']
        caption = self.captions[index]
        img_id = ann_id
        img_folder = './data/resized2014/'
        # path = os.path.join(img_folder,img_id+'.jpg')
        path = img_id+'.jpg'

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        tokens = caption
        caption = []
        # caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        caption1 = [[token] for token in caption]

        # get one-hot caption tensor
        # to_Long =  torch.Tensor(caption1).long()
        # one_hot = torch.zeros(len(caption1), 20).scatter_(1, to_Long, 1)
        # one_hot_tar = one_hot.sum(dim=0)
        # # print("one_hot_tar:", one_hot_tar)

        # get one-hot caption tensor
        # ext_eye = np.vstack((np.zeros(20), np.eye(20))) # tackle index zero as padding 
        one_hot = np.zeros(20)
        # print("caption: ", caption)
        # print("ext_eye[caption]: ", self.ext_eye[caption])
        one_hot += self.ext_eye[caption].sum(axis=0)
        # print("one_hot: ", one_hot)
        one_hot_tar = torch.Tensor(one_hot)

        caption_tensor = torch.Tensor(caption)
        return image, caption_tensor, one_hot_tar

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, one_hot_targets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  

    # Merge one_hot_targets (from tuple of 1D tensor to 2D tensor).
    o_h_target = torch.zeros(len(one_hot_targets), 20).long()
    for i, one_hot in enumerate(one_hot_targets):
        end = 20
        o_h_target[i, :end] = one_hot[:end]  

    return images, targets, o_h_target

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader