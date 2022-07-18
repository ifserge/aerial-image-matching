import numpy as np
import cv2
import os
import json
import joblib

d = joblib.load('preds.joblib')
preds = d['preds']
new_preds = d['new_preds']
statuses = d['statuses']

try:
    os.mkdir('./sub/')
except:
    pass

for i in range(len(preds)):
    fl,cx,cy,cn = preds[i]
    if statuses[i]:
        fl,cx,cy,cn = new_preds[i]
    fl = './sub/' + fl.split('/')[-1].split('.')[0] + '.json'
    
    if int(cx == 0) + int(cy == 0) + int(cx >= 10496) + (cy >= 10496) > 0:
        cx,cy = 10496 // 2, 10496 // 2

    center_x = min(max(512,int(cx)), 10496-512)
    center_y = min(max(512,int(cy)), 10496-512)
    angle = int(cn)
    
    image_center = (center_x, center_y)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    cntr = np.matmul(rot_mat, np.array([center_x, center_y, 1]))
        
    size_m = 1
    new_min_x = int(max(cntr[0]-512*size_m, 0))
    new_max_x = int(min(cntr[0]+512*size_m, 10496))
    new_min_y = int(max(cntr[1]-512*size_m, 0))
    new_max_y = int(min(cntr[1]+512*size_m, 10496))
    
    r =\
    {"left_top": [new_min_x, new_min_y], 
     "right_top": [new_max_x, new_min_y], 
     "left_bottom": [new_min_x, new_max_y], 
     "right_bottom": [new_max_x, new_max_y], 
     "angle": angle}
    with open(fl, 'w') as fp:
        json.dump(r, fp)