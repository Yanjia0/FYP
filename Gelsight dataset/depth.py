import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import dill
import calib_3d_sdk
#from ctypes import *
# either
#libc = cdll.LoadLibrary("/home/kuka-ai/FYP/calib_3d_sdk.so")
# or
#libc = CDLL("/home/kuka-ai/FYP/calib_3d_sdk.so")

def get_depth():

    path_config_file = 'R1_30bins_smoothed.pkl'
    input_folder = '/home/kuka-ai/FYP/mat1'
    #path_video_file = '/home/ruihan/Desktop/GelSight/RH_GelSight_calib_data/test_data.avi'
    init_img_path = 'mat1/mat1_1_frame_1.png'

    img_w = 320
    img_h = 240

    with open(path_config_file, "rb") as input:
        lt = dill.load(input)

    #image_list = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f[0] == '0']

    f0 = cv2.imread(init_img_path, 1)
    f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
    f0 = cv2.resize(f0, (img_w, img_h))
    fileclass = "mat"+str(1)+"/mat"+str(1)+"_"
    n=1
    for i in range(10):
        #print (str(image))
        frame = cv2.imread(fileclass+str(n+1)+"_frame_"+str(i+1)+".png")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_w, img_h))

        I = np.double(frame) - f0
        [ImGradX, ImGradY, ImGradMag, ImGradDir] = calib_3d_sdk.matchGrad(lt, I, f0)

        # get depth) map
        dm = calib_3d_sdk.poisson_reconstruct(ImGradY, ImGradX, np.zeros(frame.shape[:2]))
        print (dm.min(), dm.max(), dm.max()-dm.min())


if __name__ == '__main__':
    get_depth()