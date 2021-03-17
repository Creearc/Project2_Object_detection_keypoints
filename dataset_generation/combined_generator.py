import os
import random
import cv2
import imutils
from imutils import paths
import numpy as np
import time

import math
def rotate(origin, point, angle):
  ox, oy = origin
  px, py = point
  qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
  qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
  return int(qx), int(qy) 


def get_params(s):
  out = []
  ind = 0
  old_ind = -1
  while ind != -1:
    ind = s.find("'", ind + 1)
    if old_ind == -1:
      old_ind = ind
    else:
      out.append(int(s[old_ind + 1 : ind]))
      old_ind = -1
  return out

def make_xml(name, bbox, points, path='out'):
  f = open('{}{}.xml'.format(path, name), 'w')
  f.write('''<box top='{}' left='{}' width='{}' height='{}'>\n'''.format(bbox[1], bbox[0], bbox[2], bbox[3]))
  for i in range(len(points)):
    f.write('''<part name='{}' x='{}' y='{}'/>\n'''.format(i, points[i][0], points[i][1]))
  f.write('''</box>\n''')
  f.close()

def open_xml(name, path):
  out = []
  f = open('{}{}.xml'.format(path, name), 'r')
  box = get_params(f.readline())
  for line in f:
    out.append(get_params(line)[1:])
  f.close()
  out = out[:-1]
  return box, out

def random_img(path):
  l = os.listdir(path)
  ll = l[random.randint(0, len(l) - 1)]
  return cv2.imread('{}{}'.format(path, ll), cv2.IMREAD_UNCHANGED)

def random_position(x_l, y_l):
  if x_l[0] > x_l[1]:
    x_l = (x_l[0] - 1, x_l[0])
  if y_l[0] > y_l[1]:
    y_l = (y_l[0] - 1, y_l[0])
  return random.randint(x_l[0], x_l[1]), random.randint(y_l[0], y_l[1])

def random_size(img, s_min=0.8, s_max=1.3):
  k = random.uniform(s_min, s_max)
  out = imutils.resize(img, width = int(img.shape[1] * k))
  return out, k

def combine_imgs(img1, img2, mask, x, y):
  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  h, w = img2.shape[:2]
  x1 = x 
  x2 = x + w
  y1 = y 
  y2 = y + h
  out = img1.copy()

  alpha = mask.astype(float) / 255
  foreground = cv2.multiply(alpha, img2.astype(float))
  background = cv2.multiply(1.0 - alpha, out[y1 : y2, x1 : x2].astype(float))
  out[y1 : y2, x1 : x2] = cv2.add(foreground, background)
  return out

def combine_imgs2(img1, img2, mask, x, y):

  mask_inv = cv2.bitwise_not(mask)

  out = img1.copy()
  
  h, w = img2.shape[:2]
  x1 = x - w // 2
  x2 = x1 + w
  y1 = y - h // 2
  y2 = y1 + h
  
  roi = out[y1 : y2, x1 : x2]
  img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
  img2_fg = cv2.bitwise_and(img2, img2, mask = mask)
  
  out[y1 : y2, x1 : x2] = cv2.add(img1_bg, img2_fg)
  return out


def gen(out_img_count = 3000,
        img_dir = 'A:/Projects/Project2_Object_detection_keypoints/dataset_generation/annotated/',
        save_dir = 'out/'):
  font = cv2.FONT_HERSHEY_SIMPLEX

  backgrounds_dir = 'A:/YOLO3/generator/sortomat3/'

  # background
  b_list = os.listdir(backgrounds_dir)
  b_gamma = (0.6, 1.0) 
  b_size = (416, 416)
  b_blur = 3

  # object
  o_list = [element[:-len('.png')] for element in os.listdir(img_dir) if element[-len('.xml'):] != '.xml']
  o_gamma = (0.5, 1.5)
  o_scale = (0.5, 1.0)
  o_pos_x = (150, 150)
  o_pos_y = (100, 50)
  o_rotation = (-10, -9)
  #o_rotation = (9, 10)
  o_rotation = (-10, 10)
  n = 0

  while len(os.listdir(save_dir)) < out_img_count * 2:
    if n == len(o_list):
      n = 0
    
    out = cv2.imread('{}{}'.format(backgrounds_dir, b_list[random.randint(0, len(b_list) - 1)]))
    
    o_img = cv2.imread('{}{}.png'.format(img_dir, o_list[n]), cv2.IMREAD_UNCHANGED)
    
    o_img, k = random_size(o_img, o_scale[0], o_scale[1])
    r = random.randint(o_rotation[0], o_rotation[1])
    angle = r * np.pi/180
    o_img = imutils.rotate(o_img, r)
    o_box, o_points = open_xml(o_list[n], img_dir)
    mask = o_img[:,:,3]
    o_img = o_img[:,:,:3]

    h, w = out.shape[:2]
    o_x, o_y = random_position((o_pos_x[0], w - o_pos_x[1] - o_img.shape[1]),
                               (o_pos_y[0], h - o_pos_y[1] - o_img.shape[0]))
    
    o_box[0] = int(o_box[0] * k + o_y)
    o_box[1] = int(o_box[1] * k + o_x)
    o_box[2] = int(o_box[2] * k * 1.1)
    o_box[3] = int(o_box[3] * k * 1.3)
    o_box[0] = int(o_box[0] - o_box[3] * 0.1)
    o_box[1] = int(o_box[1] - o_box[2] * 0.05)

    c_x, c_y = o_box[1] + o_box[2] // 2, o_box[0] + o_box[3] // 2   
      
    for i in range(len(o_points)):
      o_points[i][0] = int(o_points[i][0] * k + o_x)
      o_points[i][1] = int(o_points[i][1] * k + o_y)
      o_points[i] = rotate((c_x, c_y), o_points[i], -angle)

      
    out = combine_imgs(out, o_img, mask, o_x, o_y)
    filename = time.time()
    cv2.imwrite( '{}{}.jpg'.format(save_dir, filename), out)
    make_xml(filename, o_box, o_points, path=save_dir)
    
    n += 1
    if show:
      frame = out.copy()
      cv2.circle(frame,(c_x, c_y), 3, (0,255,255), -1)
      cv2.rectangle(frame, (o_box[1], o_box[0]), (o_box[1] + o_box[2], o_box[0] + o_box[3]), (0,255,0), 1)
      for i in range(len(o_points)):
        cv2.circle(frame,(o_points[i][0], o_points[i][1]), 3, (0,0,255), -1)
        cv2.putText(frame, str(i + 1), (o_points[i][0], o_points[i][1]), font,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i + 1), (o_points[i][0], o_points[i][1]), font,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA) 
      cv2.imshow('res', frame)
      key = cv2.waitKey()
      if key == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
      

show = not True

count = 3000

t = time.time()
gen(out_img_count = count)
print((time.time() - t) / count)

