import torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def get_trainValData( k=0, spike_ready=False):
    X_train_gel = torch.FloatTensor(np.load("train.npy"))
    X_val_gel = torch.FloatTensor(np.load("val.npy"))
    y_train = torch.FloatTensor(np.load("trainlabel.npy"))
    y_val = torch.FloatTensor(np.load("vallabel.npy"))
    
    if spike_ready == False:
        return X_train_gel, y_train, X_val_gel, y_val
  
def get_testData( spike_ready=False):
    X_test_gel = torch.FloatTensor(np.load("test.npy"))
    y_test = torch.FloatTensor(np.load("testlabel.npy"))
    if spike_ready == False:
        return X_test_gel,  y_test

def get_trainValLoader( k=0, spike_ready=False, batch_size=8, shuffle=True):
    if spike_ready == False:
        X_train_gel, y_train, X_val_gel,y_val= get_trainValData( k, spike_ready)
        train_dataset = torch.utils.data.TensorDataset(X_train_gel, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)
        val_dataset = torch.utils.data.TensorDataset(X_val_gel, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size) 
        return train_loader, val_loader, train_dataset, val_dataset

def get_testLoader( spike_ready=True, batch_size=1, shuffle=True):  
    if spike_ready == False:
        X_test_gel,  y_test = get_testData( spike_ready)
        test_dataset = torch.utils.data.TensorDataset(X_test_gel, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=batch_size)
        return test_loader, test_dataset

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class channel_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        #print("inpput tensor of channel selection layer",input_tensor.shape)
        return output
class frame_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(frame_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        #print("inpput tensor of channel selection layer",input_tensor.shape)
        return output
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.selectdim = channel_selection(hidden_dim)
    def forward(self, x):
        x = self.net1(x)
        x = self.selectdim(x)
        x = self.net2(x)
        return x

class FeedForward_unpruned(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_frames, heads = 8, dim_head = 64, dropout = 0., transformerType = "space"):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.dim_head = dim_head
        #self.num_frames = num_frames
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.transformerType = transformerType
        if self.transformerType == "space":
            #pass
            self.selectqkv = channel_selection(dim_head)
        #num_frames 没有传进来
        else:
            #pass
            self.selectframe = channel_selection(num_frames+1)
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #print(qkv[1].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #print(q.shape)
        #print(self.transformerType == "temporal")
        if self.transformerType == "space":
            q = self.selectqkv(q)
            k = self.selectqkv(k)
            v = self.selectqkv(v)
            #pass
        if self.transformerType == "temporal":
            #pass
            q = rearrange(q, 'b h n d -> (b h d) n')
            q = self.selectframe(q)
            #dim_head 没有传进来/传了
            q = rearrange(q, '(b h d) n -> b h n d', b = b, n = n, d = self.dim_head)
            k = rearrange(k, 'b h n d -> (b h d) n')
           
            k = self.selectframe(k)
            k = rearrange(k, '(b h d) n -> b h n d', b = b, n = n, d = self.dim_head)
            v = rearrange(v, 'b h n d -> (b h d) n')
           
            v = self.selectframe(v)
            v = rearrange(v, '(b h d) n -> b h n d', b = b, n = n, d = self.dim_head)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        #print("this is dot result", dots.shape)
        attn = dots.softmax(dim=-1)
        #print("this is attn result", attn.shape)
        #print(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out




import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout = 0., transformerType = "space"):
        super().__init__()
        self.transformerType = transformerType
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_frames= num_frames, heads = heads, dim_head = dim_head, dropout = dropout, transformerType = transformerType)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 18, depth = 4, heads = 3, pool = 'cls', in_channels = 1, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        #print(num_patches)
        patch_dim = in_channels  * patch_height * patch_width
        #print(patch_dim)
        #print(dim)
        self.dim = dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.selectframe = frame_selection(num_frames)
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, num_frames, depth, heads, dim_head, dim*scale_dim, dropout, transformerType = "space")

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim,  num_frames, depth, heads, dim_head, dim*scale_dim, dropout, transformerType = "temporal")

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        

    def forward(self, x):
        #print("session starts, input of vivit model is ",x.shape)
        
        x = self.to_patch_embedding(x)
        #print('patch_embedding',x.shape)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        #print('cls_space_tokens',cls_space_tokens.shape)
        x = torch.cat((cls_space_tokens, x), dim=2)
        #print('x concat',x.shape)
        x += self.pos_embedding[:, :, :(n + 1)]
        #print('pos_embedding',x.shape)
        x = self.dropout(x)
        x = rearrange(x, 'b h n d -> (b n d) h')
           
        x = self.selectframe(x)
        x = rearrange(x, '(b n d) h -> b h n d', b = b, n = n+1, d = self.dim)
        x = rearrange(x, 'b t n d -> (b t) n d')
        #print('space start',x.shape)
        x = self.space_transformer(x)
        #print('space over',x.shape)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', t=t)
        #print('space to temporal x shape',x.shape)
        
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        #print('cls_temporal_tokens',cls_temporal_tokens.shape)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        #print('temporal start',x.shape)

        x = self.temporal_transformer(x)
        #print('temporal over',x.shape)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":

    #data_dir = "/home/kuka-ai/FYP/"
    kfold_number = 3 # number of fold that is chosen as the validation fold, range 0-3
    spike_ready = False # the data can be used for SNN training if the option is True
    batch_size = 1
    shuffle = True # where the data is shuffled for each epoch training 
    train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(k=kfold_number, spike_ready=False, batch_size=batch_size, shuffle=shuffle)
    test_loader, test_dataset = get_testLoader( spike_ready=False, batch_size=1, shuffle=False)

    learningrate = 0.001
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ViViT(image_size = (320,240),
              patch_size = (320,240),
              num_classes = 15,
              #num_classes = 50,
              num_frames=25,
              dim = 20, 
              depth = 4, 
              heads = 2, 
              pool = 'cls', 
              in_channels = 3, 
              dim_head = 64, 
              dropout = 0.2,
              emb_dropout = 0.3, 
              scale_dim = 2, 
             ).to(device)
    #model.load_state_dict(torch.load('modelgel_1_test0.78.pth'))
    model.load_state_dict(torch.load('model_state.pth'))
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    total = 0
    for m in model.modules():
        if isinstance(m, frame_selection):
            print(m.indexes.data)
            total += m.indexes.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, frame_selection):
            size = m.indexes.data.shape[0]
            #print(size)
            bn[index:(index+size)] = m.indexes.data.abs().clone()
            #print(m.indexes.data.abs())
            index += size

    print(bn.shape)
    y, i = torch.sort(bn)
    percent = 0.5
    thre_index = int(total * percent)
    thre = y[thre_index]
    print(thre_index)
    print(thre)
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, frame_selection):
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda(1)
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            print(pruned)
            m.indexes.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))

    pruned_ratio = pruned/total
    print("prune ratio is",pruned_ratio)
    print('Pre-processing Successful!')
    print("cfg is",cfg)


    model.eval()
    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        #for (XI, XB,  y) in test_loader:
        for i, ( XI,  y) in enumerate(test_loader):
            XI=torch.transpose(XI, 2, 3)
            XI=rearrange(XI,'b t h w c ->b t c h w ')
            x = XI
        
            data, label = x.to(device), y.long().to(device)
            #data = data.to(device)
            #label = label.to(device)

            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)

    print(f"Epoch : test_loss : {epoch_test_loss:.4f} - test_acc: {epoch_test_accuracy:.4f}\n")
    print("save pruned model",epoch_test_accuracy)
    torch.save(model.state_dict(),'model_state.pth')
    total = 0
    for m in model.modules():
        if isinstance(m, frame_selection):
            print(m.indexes.data)
            total += m.indexes.data.shape[0]
