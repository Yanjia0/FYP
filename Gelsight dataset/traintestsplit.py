import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np

if __name__ == "__main__":
    trainset = []
    for i in range(15):
        num = i+1
        loaddata = np.load("gelsight_ori_mat"+str(num)+".npy")
        print(loaddata.shape)
        trainset.append(loaddata)
    sample_np = np.array(trainset)
    trainset_np = rearrange(sample_np,'k n t h w c -> (k n) t h w c')
    print(trainset_np.shape)
    X = trainset_np
    # get label
    label=[]
    for i in range(15):
        for k in range(50):
            label.append(i)
    label_np = np.array(label)
    y = label_np
    

    #train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    with open("train_ori.npy", 'wb') as f:
            np.save(f, X_train, allow_pickle=True, fix_imports=True)
    with open("trainlabel_ori.npy", 'wb') as f:
            np.save(f, y_train, allow_pickle=True, fix_imports=True)
    #test val split
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    with open("val_ori.npy", 'wb') as f:
            np.save(f, X_train, allow_pickle=True, fix_imports=True)
    with open("vallabel_ori.npy", 'wb') as f:
            np.save(f, y_train, allow_pickle=True, fix_imports=True)
    with open("test_ori.npy", 'wb') as f:
            np.save(f, X_val, allow_pickle=True, fix_imports=True)
    with open("testlabel_ori.npy", 'wb') as f:
            np.save(f, y_val, allow_pickle=True, fix_imports=True)