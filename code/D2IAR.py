import cv2
import numpy as np
import scipy.io as scio
import cv2
import cv2
import matplotlib.pyplot as plt
import spectral as spr
import math
from queue import Queue
import math
import scipy.stats
# from sklearn.neighbors import kde
import scipy.stats as st
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
#dataset load
# img_2006 = scio.loadmat(r'./PreImg_2006_new.mat')['img']
# img_2007 = scio.loadmat(r'./PostImg_2007_new.mat')['img']
#for farmland datasets, selected bands are 25 75 120
#for hermiston datasets, selected bands are 44 50 92
#for river datasets, selected bands are 58 76 82
#for bay_area datasets, selected bands are 35 46 58

img_20061 = scio.loadmat(r'D:/RS_dataset/ChangeDetectionDataset-master/Hermiston/hermiston2004.mat')['HypeRvieW']
img_20071 = scio.loadmat(r'D:/RS_dataset/ChangeDetectionDataset-master/Hermiston/hermiston2007.mat')['HypeRvieW']
row,col,band = img_20061.shape
img_2006 = np.zeros((row,col,3))
img_2007 = np.zeros((row,col,3))
img_2006[:,:,0] = img_20061[:,:,44]
img_2006[:,:,1] = img_20061[:,:,50]
img_2006[:,:,2] = img_20061[:,:,92]
img_2007[:,:,0] = img_20071[:,:,44]
img_2007[:,:,1] = img_20071[:,:,50]
img_2007[:,:,2] = img_20071[:,:,92]


row,col,band = img_2006.shape
img_2006=np.float64(img_2006)
img_2007=np.float64(img_2007)
for i in range(band):
    img_2006[:, :, i] = (img_2006[:, :, i] - np.min(img_2006[:, :, i])) / (np.max(img_2006[:, :, i]) - np.min(img_2006[:, :, i])) * 255
    img_2006[:, :, i] = np.array(img_2006[:, :, i])
    img_2006 = img_2006.astype(int)

    img_2007[:, :, i] = (img_2007[:, :, i] - np.min(img_2007[:, :, i])) / (np.max(img_2007[:, :, i]) - np.min(img_2007[:, :, i])) * 255
    img_2007[:, :, i] = np.array(img_2007[:, :, i])
    img_2007 = img_2007.astype(int)

T1=20
T2=85
def get_pixels(img,index_list_arr,band):
    pixels = []
    for i,t in enumerate(index_list_arr):
        pixels.append(img[(*t,band)])
    return pixels

def adaptive_region(img,x,y,channal,t1=T1,t2=T2):#adaptive region of expansion for pixel(x,y)
    gray = int(img[x, y, channal])
    all_index = []
    q =Queue()
    q.put((x,y))
    x1= x
    y1=y
    pixel=[]
    while  not q.empty():
        x,y = q.get()
        for i in range(max(0,x-1),min(row,x+2)):
            for j  in range(max(0,y-1),min(col,y+2)):
                if(i==x1 and j==y1):continue
                if (    abs(int( img[i,j,channal]-  gray))    <= t1 and t2 > 0):
                    if (i, j) not in all_index:
                        t2 -= 1
                        all_index.append((i, j))
                        pixel.append(img[i,j,channal])
                        q.put((i,j))
    all_index.append((x1,y1))
    pixel.append(img[x1,y1,channal])
    return all_index,pixel


def get_range_90(arr, account=0.8):#get the information from the distrubution
    max_index = np.argmax(arr)
    i = 0
    count = arr[max_index]
    range_left = max_index
    range_right = max_index
    while (count < account and count < sum(arr)):
        flg = count
        if (range_right < 255):
            range_right += 1
            count += arr[range_right]
        if (range_left > 0):
            range_left -= 1
            count += arr[range_left]
        if (flg == count):
            break
    return range(range_left, range_right, 1)

entropy_DI = np.zeros((row, col, band))#generation of distance in image1
entropy_DI1 = np.zeros((row, col, band))#generation of distance in image2
for b in range(band):
    print("band:", b)
    for i in range(row):
        print('row:',i)
        for j in range(col):
            pixel_before = img_2006[i, j, b]
            pixel_after = img_2007[i, j, b]
            area_before, before = adaptive_region(img_2006, i, j, b, T1, T2)
            area_after, after = adaptive_region(img_2007, i, j, b, T1, T2)
            before = np.array(before)
            after = np.array(after)
            bandwidth = 10
            model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            x_range = np.linspace(0, 255, 256)
            model.fit(after[:, np.newaxis])
            x_log_prob1 = model.score_samples(x_range[:, np.newaxis])
            x_prob1 = np.exp(x_log_prob1)
            range2 = get_range_90(x_prob1)
            before_after_cross = get_pixels(img_2006, area_after, b)
            before_after_cross = np.array(before_after_cross)
            model.fit(before_after_cross[:, np.newaxis])
            x_range = np.linspace(0, 255, 256)
            x_log_prob = model.score_samples(x_range[:, np.newaxis])
            x_prob = np.exp(x_log_prob)
            range1 = get_range_90(x_prob)
            model.fit(after[:, np.newaxis])
            x_log_prob1 = model.score_samples(x_range[:, np.newaxis])
            x_prob1 = np.exp(x_log_prob1)
            range2 = get_range_90(x_prob1)
            dist1 = abs((x_prob[range1] * x_range[range1]).sum() - (x_prob1[range2] * x_range[range2]).sum())
            after_before_cross = get_pixels(img_2007, area_before, b)
            after_before_cross = np.array(after_before_cross)
            bandwidth = 10
            model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            model.fit(after_before_cross[:, np.newaxis])
            x_range = np.linspace(0, 255, 256)
            # x_range = np.linspace(0, 255,255)
            x_log_prob = model.score_samples(x_range[:, np.newaxis])
            x_prob = np.exp(x_log_prob)
            range1 = get_range_90(x_prob)
            model.fit(before[:, np.newaxis])
            x_log_prob1 = model.score_samples(x_range[:, np.newaxis])
            x_prob1 = np.exp(x_log_prob1)
            range2 = get_range_90(x_prob1)
            dist2 = abs((x_prob[range1] * x_range[range1]).sum() - (x_prob1[range2] * x_range[range2]).sum())
            entropy_DI[i, j, b] = (dist1 + dist2) / 2

for i in range(band):
    entropy_DI[:, :, i] = (entropy_DI[:, :, i] - np.min(entropy_DI[:, :, i])) / (np.max(entropy_DI[:, :, i]) - np.min(entropy_DI[:, :, i]))
    entropy_DI[:, :, i] = entropy_DI[:, :, i] * 255
entropy_DI = np.uint8(entropy_DI)
scio.savemat(r"D:\text_all\test_hemiston",{"HypeRvieW":entropy_DI_test_t1})