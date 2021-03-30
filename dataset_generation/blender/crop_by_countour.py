import cv2
import os
import pandas as pd
import imutils
import random

img_dir = 'img_kp\\' # object dirrectory
bkg_dir = 'bkg\\' # background dirrectory
newimg_dir = 'img_kp_cropped\\'
ext = '.png'
newimgpath = 'D:\\LumRec\\YOLO_and_Dlib_KPD\\kp_dataset_generation\\img_kp_cropped\\'

i = 0
for file in os.listdir(img_dir):

    a = os.listdir(bkg_dir)
    bkg = a[random.randint(0, len(a)-1)]

    lumimgname = file
    newbkgimgname = bkg
    print(str(i)+'. '+lumimgname)
    print(str(i)+'. '+newbkgimgname)
    print('  ')
    i += 1

    lumimg = cv2.imread(img_dir+lumimgname)
    newbkgimg = cv2.imread(bkg_dir+newbkgimgname)

    newbkgimg = imutils.resize(newbkgimg, height = 800)

    rows,cols,channels = lumimg.shape
    roi = newbkgimg[0:rows, 0:cols]

    lum = lumimg
       
    gray = cv2.cvtColor(lum, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    xmin = cols
    xmax = 0
    ymin = rows
    ymax = 0

    for h in range(cols): 
        for w in range(rows):
            if mask[w,h] == 255:
                if h < xmin:
                    xmin = h
                if h > xmax:
                    xmax = h
                if w < ymin:
                    ymin = w
                if w > ymax: 
                    ymax = w

    new1 = cv2.bitwise_and(roi, roi, mask = mask_inv)
    new2 = cv2.bitwise_and(lumimg, lumimg, mask = mask)

    dst = cv2.add(new1,new2)
    #newbkgimg[0:rows, 0:cols ] = dst # with background
    newbkgimg[0:rows, 0:cols ] = new2 # without background
    #newbkgimg[ymin:ymax+1, xmin:xmax+1] = cv2.medianBlur(newbkgimg[ymin:ymax+1,xmin:xmax+1],3) # with background
    newbkgimg = newbkgimg[ymin:ymax+1, xmin:xmax+1]
    cv2.imwrite(newimgpath+lumimgname[:-4]+ext, newbkgimg)

cv2.waitKey(0)
cv2.destroyAllWindows()