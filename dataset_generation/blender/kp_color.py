import cv2
#import os
import pandas as pd
import numpy as np

lumimgname = "0099.png"

lumimg = cv2.imread(lumimgname)
cols,rows,channels = lumimg.shape

p1_hsv_min = np.array((140,204,204), np.uint8) # pink
p1_hsv_max = np.array((153, 230, 255), np.uint8)
p2_hsv_min = np.array((102,230,230), np.uint8) # blue
p2_hsv_max = np.array((128, 255, 255), np.uint8)
p3_hsv_min = np.array((45,230,230), np.uint8) # green
p3_hsv_max = np.array((68, 255, 255), np.uint8)
p4_hsv_min = np.array((1,230,230), np.uint8) # red
p4_hsv_max = np.array((10, 255, 255), np.uint8)
p5_hsv_min = np.array((25,230,230), np.uint8) # yellow
p5_hsv_max = np.array((50, 255, 255), np.uint8)
p6_hsv_min = np.array((77,230,230), np.uint8) #cyan
p6_hsv_max = np.array((100, 255, 255), np.uint8)
p7_hsv_min = np.array((10,230,230), np.uint8) # orange
p7_hsv_max = np.array((20, 255, 255), np.uint8)

hsv = cv2.cvtColor(lumimg, cv2.COLOR_BGR2HSV)
p1_threshold = cv2.inRange(hsv, p1_hsv_min, p1_hsv_max)
p2_threshold = cv2.inRange(hsv, p2_hsv_min, p2_hsv_max)
p3_threshold = cv2.inRange(hsv, p3_hsv_min, p3_hsv_max)
p4_threshold = cv2.inRange(hsv, p4_hsv_min, p4_hsv_max)
p5_threshold = cv2.inRange(hsv, p5_hsv_min, p5_hsv_max)
p6_threshold = cv2.inRange(hsv, p6_hsv_min, p6_hsv_max)
p7_threshold = cv2.inRange(hsv, p7_hsv_min, p7_hsv_max)
#p1_threshold = cv2.bitwise_not(p1_threshold)

samp1 = cv2.bitwise_and(lumimg, lumimg, mask = p1_threshold)
samp2 = cv2.bitwise_and(lumimg, lumimg, mask = p2_threshold)
samp3 = cv2.bitwise_and(lumimg, lumimg, mask = p3_threshold)
samp4 = cv2.bitwise_and(lumimg, lumimg, mask = p4_threshold)
samp5 = cv2.bitwise_and(lumimg, lumimg, mask = p5_threshold)
samp6 = cv2.bitwise_and(lumimg, lumimg, mask = p6_threshold)
samp7 = cv2.bitwise_and(lumimg, lumimg, mask = p7_threshold)

sample = samp1 + samp2 + samp3 + samp4 + samp5 + samp6 + samp7

# Сделать массивы с threshold для каждой точки

xmin = rows
xmax = 0
ymin = cols
ymax = 0

# Далее в цикле по элементам массива сразу записывать найденные координаты 
# в форму xml чтобы потом её сохранить
'''
for h in range(rows): # Сделать с поиском контуров чтобы не просматривать каждый пиксель
    for w in range(cols):
        if mask[w,h] == 255:
            if h < xmin:
                xmin = h
            if h > xmax:
                xmax = h
            if w < ymin:
                ymin = w
            if w > ymax: 
                ymax = w

'''
cv2.imshow("new0", lumimg)
cv2.imshow("new1", sample)

cv2.waitKey(0)
cv2.destroyAllWindows()