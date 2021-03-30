"""
python kp_det_evaluation.py --shape-predictor kplum_gen_third_comb_test.dat
"""

from imutils import face_utils
import argparse
import imutils
import pandas as pd
import time
import dlib
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

def get_kp_from_xml(xmlname):
    in_file = open(xmlname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    name = []
    x = []
    y = []
    for i in range(0, len(root)):
        name.append(root[i].attrib['name'])
        x.append(root[i].attrib['x'])
        y.append(root[i].attrib['y'])
    return name, x, y


def kp_distance(kp_true, kp_pred):
    from math import sqrt
    kp_var = []
    for i in range(len(kp_true)):
        kp_var.append(sqrt((kp_true[i][0]-kp_pred[i][0])**2 + (kp_true[i][1]-kp_pred[i][1])**2))
    return kp_var

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

print(args["shape_predictor"])
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(args["shape_predictor"])
j = 0

img_path = '_third_combination_test/'
ext = '.jpg'

fls = os.listdir(img_path)
font = cv2.FONT_HERSHEY_SIMPLEX

result = {'filename':[], 'dist0':[], 'dist1':[], 'dist2':[], 'dist3':[], 'dist4':[], 'time, sec': []}
resultList = pd.DataFrame(result)

for file in fls:
    startTime = time.time()

    imgname = file
    
    if not (img_path + imgname).endswith(ext):
        continue
    else:
        frame = cv2.imread(img_path + '/' + imgname)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        print(str(j) + ') ' + imgname)    
        xmlname = imgname[:-4] + '.xml'
        name, x, y = get_kp_from_xml(img_path + xmlname)
        true_kp = []
        for i in range(len(name)):
            true_kp.append([int(x[i]),int(y[i])])
        true_kp = np.array(true_kp)
        rect = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        dist = kp_distance(true_kp, shape)
        recTime = time.time() - startTime
        new_row = pd.DataFrame({'filename':imgname, 'dist0':dist[0], 'dist1':dist[1], 'dist2':dist[2], 'dist3':dist[3], 'dist4':dist[4], 'time, sec': recTime}, index = [j])
        resultList = pd.concat([resultList, new_row]).reset_index(drop = True)
        j += 1  

    if not True:
        if j == 3:
            break


print(resultList)
print("kp_eval_{0}.xlsx".format(str(args["shape_predictor"])[10:-4]))
resultList.to_excel("kp_eval_{0}_2.xlsx".format(str(args["shape_predictor"])[10:-4]))