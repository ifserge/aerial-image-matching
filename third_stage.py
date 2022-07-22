import pandas as pd 
import numpy as np
import glob
from tqdm import tqdm
import cv2
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

from math import sin, cos
from asift import Timer, image_resize, init_feature, filter_matches, affine_detect
import joblib
from mpire import WorkerPool
from multiprocessing.pool import ThreadPool

json_dir = "./train/json/"

data_df = pd.DataFrame({'id': [], "left_top_x": [], 'left_top_y': [], "right_bottom_x": [], 'right_bottom_y': [], 'angle': []})

json_true = []
for _, _, files in os.walk(json_dir):
    for x in files:
        if x.endswith(".json"):
            data = json.load(open(json_dir + x))
            new_row = {'id':x.split(".")[0]+".img", 'left_top_x':data["left_top"][0], 'left_top_y':data["left_top"][1], 'right_bottom_x': data["right_bottom"][0], "right_bottom_y": data["right_bottom"][1], 'angle': data["angle"]}
            data_df = data_df.append(new_row, ignore_index=True)

            
data_df['width'] = (data_df['right_bottom_x'] - data_df['left_top_x']).abs()
data_df['height'] = (data_df['right_bottom_y'] - data_df['left_top_y']).abs()
data_df['center_x'] = ((data_df['right_bottom_x'] + data_df['left_top_x']) * 0.5).astype(int)
data_df['center_y'] = ((data_df['right_bottom_y'] + data_df['left_top_y']) * 0.5).astype(int)
data_df['id'] = data_df['id'].apply(lambda s: './train/img/' + s.split('.')[0]+'.png')
data_df[['width','height','angle']].describe()

original = np.array(Image.open('./original.tiff'))

def rotate_image(image, point, angle):
    rot_mat = cv2.getRotationMatrix2D(point, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_fragment(center_x, center_y, angle, size_m=1):
    image_center = (center_x, center_y)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    cntr = np.matmul(rot_mat, np.array([center_x, center_y, 1]))
    
    fragment = rotate_image(original, image_center, angle)
    
    new_min_x = max(cntr[0]-512*size_m, 0)
    new_max_x = min(cntr[0]+512*size_m, 10496)
    new_min_y = max(cntr[1]-512*size_m, 0)
    new_max_y = min(cntr[1]+512*size_m, 10496)

    return fragment[int(new_min_y):int(new_max_y), int(new_min_x):int(new_max_x)]

detector_name = "sift-flann"
detector, matcher = init_feature(detector_name)

train_dict,test_dict = {},{}
clahe = cv2.createCLAHE(clipLimit=16, tileGridSize=(16,16))
with ThreadPool(processes=cv2.getNumberOfCPUs()) as pool:
    for fl in data_df['id'].values:
        ori_img1 = np.array(Image.open(fl))   
        img1 = cv2.cvtColor(ori_img1, cv2.COLOR_RGB2GRAY)
        kp, desc = affine_detect(detector, img1, pool=pool)
        if len(kp) > 10000:
            continue
        
        img1 = clahe.apply(img1)
        kp, desc = affine_detect(detector, img1, pool=pool)
        train_dict[fl] = ([p.pt for p in kp], desc)
        print(f'{fl} passed with {len(kp)} points')
        
    print(len(train_dict))
    
def filter_matches2(kp1, kp2, matches, ratio=0.7):
    mkp1, mkp2 = [], []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])

    p1 = np.array(mkp1)
    p2 = np.array(mkp2)
    kp_pairs = zip(mkp1, mkp2)

    return p1, p2, list(kp_pairs)

def m_match(matcher, train_img, kp1, desc1, kp2, desc2):
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kp_pairs = filter_matches2(kp1, kp2, raw_matches, ratio=0.7)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 100.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        return (train_img, len(kp_pairs))
    return (train_img, 0)

d = joblib.load('preds.joblib')
preds = d['preds']
new_preds = d['new_preds']
statuses = d['statuses']

for i_img,(fl, x, y, a) in enumerate(preds):
    if statuses[i_img]:
        continue
        
    ori_img1 = np.array(Image.open(fl))   
    img1 = cv2.cvtColor(ori_img1, cv2.COLOR_RGB2GRAY)
    img1 = clahe.apply(img1)
    with ThreadPool(processes=16) as pool:
        kp1, desc1 = affine_detect(detector, img1, pool=pool)
    pool.close()

    print(f'processing img {i_img} with {len(kp1)} keypoints...')
    

    kp1 = [p.pt for p in kp1]
    candidates = []
    for train_img, (kp2, desc2) in train_dict.items():
        candidates.append((train_img, kp2, desc2))
            
    with WorkerPool(n_jobs=20, shared_objects=(matcher, kp1, desc1)) as pool:
        results = pool.map(m_match, candidates)
        
    try:
        pool.close()
    except:
        pass
        
    matches = sorted(results, key=lambda x: x[1], reverse=True)
    print('max matches: ', matches[0])
    
    if (matches[0][1] > 100) or (matches[0][1] > len(kp1) * 0.5):
        MAX_SIZE = 1024
        ori_img1_ = np.array(Image.open(preds[i_img][0]))
        ori_img1_ = cv2.cvtColor(ori_img1_, cv2.COLOR_RGB2GRAY)
        ori_img1_ = clahe.apply(ori_img1_)
        ori_img2_ = np.array(Image.open(matches[0][0]))
        x,y,a = data_df.loc[data_df['id'] == matches[0][0],['center_x','center_y','angle']].values[0]
        ori_img2_ = cv2.cvtColor(ori_img2_, cv2.COLOR_RGB2GRAY)
        ori_img2_ = clahe.apply(ori_img2_)

        ratio_1 = 1
        ratio_2 = 1

        if ori_img1_.shape[0] > MAX_SIZE or ori_img1_.shape[1] > MAX_SIZE:
            ratio_1 = MAX_SIZE / ori_img1_.shape[1]
            print("Large input detected, image 1 will be resized")
            img1_ = image_resize(ori_img1_, ratio_1)
        else:
            img1_ = ori_img1_

        if ori_img2_.shape[0] > MAX_SIZE or ori_img2_.shape[1] > MAX_SIZE:
            ratio_2 = MAX_SIZE / ori_img2_.shape[1]
            print("Large input detected, image 2 will be resized")
            img2_ = image_resize(ori_img2_, ratio_2)
        else:
            img2_ = ori_img2_

        print(f"Using {detector_name.upper()} detector...")

        # Profile time consumption of keypoints extraction
        with Timer(f"Extracting {detector_name.upper()} keypoints..."):
            pool = ThreadPool(processes=cv2.getNumberOfCPUs())
            kp1, desc1 = affine_detect(detector, img1_, pool=pool)
            kp2, desc2 = affine_detect(detector, img2_, pool=pool)

        print(f"img1 - {len(kp1)} features, img2 - {len(kp2)} features")

        # Profile time consumption of keypoints matching
        with Timer('Matching...'):
            raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=0.7)

        for index in range(len(p1)):
            pt = p1[index]
            p1[index] = pt / ratio_1

        for index in range(len(p2)):
            pt = p2[index]
            p2[index] = pt / ratio_2

        for index in range(len(kp_pairs)):
            element = kp_pairs[index]
            kp1, kp2 = element

            new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
            new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

            kp_pairs[index] = (new_kp1, new_kp2)

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 8.0)
        print(f"{np.sum(status)} / {len(status)}  inliers/matched")
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]     
        h1, w1 = img1_.shape[:2]
        h2, w2 = img2_.shape[:2]

        if H is not None:
            corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
            corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            new_cx = corners[0][0]*0.5 + corners[2][0]*0.5 - w1
            new_cy = corners[0][1]*0.5 + corners[2][1]*0.5
            new_cx2 = corners[1][0]*0.5 + corners[2][0]*0.5 - w1
            new_cy2 = corners[1][1]*0.5 + corners[2][1]*0.5
            rot_mat = cv2.getRotationMatrix2D((w2*0.5, h2*0.5), -a, 1.0)
            cntr = np.matmul(rot_mat, np.array([new_cx, new_cy, 1])) 
            cntr2 = np.matmul(rot_mat, np.array([new_cx2, new_cy2, 1])) 
            da = math.atan2(cntr2[1] - cntr[1], cntr2[0] - cntr[0]) / math.pi * 180
            if da < 360:
                da += 360
            if da > 360:
                da -= 360
            xn1,yn1,an1 = int(x) + cntr[0] - w2*0.5, int(y) + cntr[1] - h2*0.5, da
            
            new_preds[i_img] = (preds[i_img][0],xn1,yn1,an1)
            statuses[i_img] = True
            print(new_preds[i_img])
            d = joblib.load('preds.joblib')
            d['new_preds'] = new_preds
            d['statuses'] = statuses
            joblib.dump(d, 'preds.joblib')
