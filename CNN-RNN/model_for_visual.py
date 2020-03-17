import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import avg_pool2d
from torch.autograd import Variable
from torchsummary import summary

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.features[i] for i in range(29))      # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg(images)
        N,C,H,W=features.size()
        #print('features',features.size())
        features = features.view(N,C,H*W)
        features = features.permute(0,2,1)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=10):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size*2,hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size= vocab_size
        self.vis_dim=512
        self.hidden_dim=1024
        vis_num=196
        self.att_vw = nn.Linear(self.vis_dim,self.vis_dim,bias=False)
        self.att_hw = nn.Linear(self.hidden_dim,self.vis_dim,bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim,1,bias=False)

    def attention(self,features,hiddens):
        att_fea = self.att_vw(features)     #(80, 196, 512)
        att_h = self.att_hw(hiddens).unsqueeze(1)   #(80,1,512)
        att_full = nn.ReLU()(att_fea + att_h +self.att_bias.view(1,-1,1)) # bias.view(1,-1,1) (1,196,1)
        att_out = self.att_w(att_full)
        alpha=nn.Softmax(dim=1)(att_out)    # (80, 196, 1)
        fea_alp=features*alpha
        context=torch.sum(fea_alp,1) # features (1, 196, 512)
        # context shape [80, 512]
        return context,alpha,fea_alp

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # print("feature.shape", features.shape) # (80, 196, 512)
        # print("caption.shape", captions.shape) # (80, 5) 5 is not fixed
        # print("lengths.shape", len(lengths)) # list (80)
        embeddings = self.embed(captions)   # (80, 6, 512)
        feats=torch.mean(features,1).unsqueeze(1)   # (80, 1, 512)
        embeddings = torch.cat((feats, embeddings), 1)  # (80, 7, 512)
        batch_size, time_step = captions.size()
        predicts = to_var(torch.zeros(batch_size, time_step, self.vocab_size))
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hx=to_var(torch.zeros(batch_size, 1024))
        cx=to_var(torch.zeros(batch_size, 1024))
        for i in range(time_step): 
            feas,_=self.attention(features,hx)
            input=torch.cat((feas,embeddings[:,i,:]),-1)
            hx,cx = self.lstm_cell(input,(hx,cx))
            output = self.linear(hx)
            predicts[:,i,:]=output
        return predicts
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        hx=to_var(torch.zeros(1,1024))
        cx=to_var(torch.zeros(1,1024))
        inputs = torch.mean(features,1)
        alphas=[]
        feas_= features
        for i in range(self.max_seg_length):
            feas,alpha,fea_alp=self.attention(features,hx)
            feas_=torch.cat((feas_,fea_alp), 0)
            alphas.append(alpha)
            inputs=torch.cat((feas,inputs),-1)
            hx, cx = self.lstm_cell(inputs,(hx,cx))          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hx.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        feas_=torch.sum(feas_,2)
        return sampled_ids,alphas,feas_
        
# encoder = EncoderCNN(512)
# summary(encoder, (3, 224, 224))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# decoder = DecoderRNN(512, 1024, 2, 1, 40).to(device)
# # print(decoder)
# summary(decoder, [(128,196,512,), (128,128,), (128,)])
