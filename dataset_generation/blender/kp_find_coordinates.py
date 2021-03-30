import cv2
import os
import pandas as pd
import numpy as np

def make_xml(name, bbox, points, path='out'):
  f = open('{}/{}.xml'.format(path, name), 'w')
  f.write('''      <box top='{}' left='{}' width='{}' height='{}'>\n'''.format(bbox[0], bbox[1], bbox[2], bbox[3]))
  for i in range(len(points)):
    f.write('''        <part name='{}' x='{}' y='{}'/>\n'''.format(i, points[i][0], points[i][1]))
  f.write('''      </box>\n''')
  f.close()

img_dir = 'img_kp_cropped\\'
save_path = 'img_cropped_with_bkg'

for file in os.listdir(img_dir):
    lumimgname = file
    print(lumimgname)
    lumimg = cv2.imread(img_dir+lumimgname)
    #cv2.imshow("lumimg", lumimg)

    cols,rows,channels = lumimg.shape
    point_number = 5

    # 0 - pink; 1 - blue; 2 - red; 3 - cyan; 4 - orange; 5 - green;
    hsv_colours = np.array([[[140,204,204], [153, 230, 255]], 
                            [[102,230,230], [128, 255, 255]],
                            [[1,230,230],   [10,  255, 255]],
                            [[77,230,230],  [100, 255, 255]],
                            [[10,230,230],  [20, 255,  255]],
                            [[45,230,230],  [68, 255,  255]]])

    points = []

    hsv = cv2.cvtColor(lumimg, cv2.COLOR_BGR2HSV)

    for i in range(point_number):
        points.append(cv2.inRange(hsv, hsv_colours[i,0], hsv_colours[i,1]))

    pc = []

    for i in range(point_number):
        xmin = rows
        xmax = 0
        ymin = cols
        ymax = 0
        for h in range(rows):
            for w in range(cols):
                if points[i][w,h] == 255:
                    if h < xmin:
                        xmin = h
                    if h > xmax:
                        xmax = h
                    if w < ymin:
                        ymin = w
                    if w > ymax: 
                        ymax = w

        pc.append(((xmax-xmin)//2 + xmin, (ymax-ymin)//2 + ymin))

    print(pc)
    make_xml(file[:-4], [0,0,lumimg.shape[1],lumimg.shape[0]], pc, path=save_path)
    

cv2.waitKey(0)
cv2.destroyAllWindows()