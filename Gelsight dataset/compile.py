import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps
def subtract(a,b):
     return  "".join(a.rsplit(b))

if __name__ == "__main__":

    dirc = "/home/kuka-ai/FYP/"
    for c in range(1,16):
        oneclass = []
        for i in range (1,51):
            if c==3 and i==3:
                continue
            if c==6 and i==19:
                continue
            if c==7 and i==19:
                continue
            if c==8 and i==18:
                continue
            f = open(str(dirc)+"mat"+str(c)+"/mat"+str(c)+"_"+str(i)+".txt")
            time = f.read().splitlines()
            f.close()
            matData = pd.read_csv(str(dirc)+"mat"+str(c)+"/mat"+str(c)+"_"+str(i)+".csv",dtype = "string")
            df = matData.loc[(matData['Timestamps']>= time[2])&(matData['Timestamps']<=time[4])]
            print(df.iloc[0,1])
            path = subtract(df.iloc[0,1],'/home/tas/Desktop/historical/processed/GelSight/')
            #savedirc = "/home/kuka-ai/FYP/gelsight"
            sample = []
            for ind in range(25):
                path = subtract(df.iloc[ind,1],'/home/tas/Desktop/historical/processed/GelSight/')
                image = Image.open(dirc+path)
                if c==12:
                    zero = Image.open("mat"+str(c)+"/mat"+str(c)+"_"+"1_frame_3.png")
                if c==13:
                    zero = Image.open("mat"+str(c)+"/mat"+str(c)+"_"+"1_frame_6.png")
                if c==14:
                    zero = Image.open("mat"+str(c)+"/mat"+str(c)+"_"+"1_frame_10.png")
                if c==15:
                    zero = Image.open("mat"+str(c)+"/mat"+str(c)+"_"+"1_frame_6.png")
                else:
                    zero = Image.open("mat"+str(c)+"/mat"+str(c)+"_"+"1_frame_1.png")
                image_arr = np.array(image) - np.array(zero)
                image_arr = image_arr[100:340,150:470,:]
                sample.append(image_arr)
            sample_np = np.array(sample)
            if c==3 and i==2:
                oneclass.append(sample_np)
            if c==6 and i==2:
                oneclass.append(sample_np)
            if c==7 and i==2:
                oneclass.append(sample_np)
            if c==8 and i==2:
                oneclass.append(sample_np)
            oneclass.append(sample_np)
            oneclass_np = np.array(oneclass)
            print(oneclass_np.shape)
            with open('gelsight_ori_mat'+str(c)+'.npy','wb') as f:
                np.save(f,oneclass_np,allow_pickle=True, fix_imports = True)
    
