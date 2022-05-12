import torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# create utilis here
#def get_ohe(_Y, num_class = 20):

#    target_class = np.zeros([_Y.shape[0], num_class])

#    for i in range(target_class.shape[0]):

#        target_class[i, int(_Y[i])] = 1

#    return target_class

def get_trainValData( k=0, spike_ready=False):
    #num_class = 20
    #bio_len = 150

    # read data
    #X_train_gel = torch.FloatTensor(np.load("train_3class.npy"))
    X_train_gel = torch.FloatTensor(np.load("train.npy"))
    #X_train_gel = rearrange(X_train_gel, 'i b h n d -> (i b) h n d')
    #X_val_gel = torch.FloatTensor(np.load("val_3class.npy"))
    X_val_gel = torch.FloatTensor(np.load("val.npy"))
    #X_val_gel = rearrange(X_val_gel, 'i b h n d -> (i b) h n d')
    #y_train = torch.FloatTensor(np.load("train_label.npy"))
    #y_val = torch.FloatTensor(np.load("val_label.npy"))
    y_train = torch.FloatTensor(np.load("trainlabel.npy"))
    y_val = torch.FloatTensor(np.load("vallabel.npy"))
    
    if spike_ready == False:
        return X_train_gel, y_train, X_val_gel, y_val

        
   
def get_testData( spike_ready=False):
    #num_class = 20
    #bio_len = 150

    X_test_gel = torch.FloatTensor(np.load("test.npy"))
    #X_test_gel = rearrange(X_test_gel, 'i b h n d -> (i b) h n d')
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
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
        print(num_patches)
        patch_dim = in_channels  * patch_height * patch_width
        print(patch_dim)
        print(dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        #print('1111')
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

learningrate = 0.0003
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
              heads = 4, 
              pool = 'cls', 
              in_channels = 3, 
              dim_head = 64, 
              dropout = 0.2,
              emb_dropout = 0.3, 
              scale_dim = 2, 
             ).to(device)
    #model.load_state_dict(torch.load('model_state.pth'))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    best_val = 0
    train_acc = []
    val_acc = []
    

    #TEST MODEL
    model.load_state_dict(torch.load('1patch_head4_dim20_2scale_FINETUNE0.9.pth'))
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