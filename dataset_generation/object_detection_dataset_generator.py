import os
import random
import cv2
import imutils
from imutils import paths
import numpy as np
import pandas as pd
import time

def get_contours(img):
  cnts, hierarchy = cv2.findContours(img,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
  boxes = []
  for cnt in cnts:
    rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    x1 = min(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    x2 = max(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    y1 = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    y2 = max(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    boxes.append(((x1, y1), (x2, y2))) 
  return boxes

def crop_by_countour(img, cnt):
  x1, y1, x2, y2 = cnt[0][0], cnt[0][1], cnt[1][0], cnt[1][1]
  return img[y1 : y2, x1 : x2]

def get_mask(img, thr1=0, thr2=255):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(gray, thr1, 255, cv2.THRESH_BINARY)
  ret, mask2 = cv2.threshold(gray, thr2, 255, cv2.THRESH_BINARY_INV)
  return cv2.bitwise_and(mask, mask2)

def adjust_gamma(img, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(img, table)

def random_img(path):
  l = os.listdir(path)
  ll = l[random.randint(0, len(l) - 1)]
  return cv2.imread('{}{}'.format(path, ll), cv2.IMREAD_UNCHANGED)

def random_size(img, s_min=0.8, s_max=1.3):
  out = imutils.resize(img, width = int(img.shape[1] * random.uniform(s_min, s_max)))
  return out

def random_position(x_l, y_l):
  return random.randint(x_l[0], x_l[1]), random.randint(y_l[0], y_l[1])

def combine_imgs(img1, img2, mask, x, y):
  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  h1, w1 = img1.shape[:2]
  h2, w2 = img2.shape[:2]
  x11, x12 = np.clip(x - w2 // 2, 0, w1 - 1), np.clip(x + w2 // 2, 0, w1 - 1)
  y11, y12 = np.clip(y - h2 // 2, 0, h1 - 1), np.clip(y + h2 // 2, 0, h1 - 1)
  x21 = x11 - (x - w2 // 2)
  y21 = y11 - (y - h2 // 2)
  x22 = np.clip(x21 + x12 - x11, 0, w2)
  y22 = np.clip(y21 + y12 - y11, 0, h2)
  out = img1.copy()

  alpha = mask[y21 : y22, x21 : x22].astype(float) / 255
  foreground = cv2.multiply(alpha, img2[y21 : y22, x21 : x22].astype(float))
  background = cv2.multiply(1.0 - alpha, out[y11 : y12, x11 : x12].astype(float))
  out[y11 : y12, x11 : x12] = cv2.add(foreground, background)
  return out

def make_xml(filename, cnt, ext):
  x1, y1, x2, y2 = cnt[0][0], cnt[0][1], cnt[1][0], cnt[1][1]
  with open('template_lum_bbox.xml') as temp:
    tmp = temp.read()
    xml = tmp.format('{}.{}'.format(filename, ext), '{}\\{}.{}'.format(save_dir, filename,ext), b_size[0], b_size[1], 3, x1, y1, x2, y2)
    new_xml = open('{}\\{}.xml'.format(save_dir, filename), 'w') 
    new_xml.write(xml)
    new_xml.close()

seed = 1234
out_img_count = 40000
img_dir = 'lum_median/'
backgrounds_dir = 'backgrounds/'
branches_dir = 'branches/'
save_dir = 'test_saves/'
ext = 'jpg'

# background
b_gamma = (0.7, 2.4)
b_size = (416, 416)
b_blur = 5
scale_coef = 4
b_mega_size = (scale_coef * b_size[0], scale_coef * b_size[1])


# luminare
o_gamma = (0.5, 2.4)
o_scale = (0.7, 2.7)
o_pos_x = (250, b_mega_size[0]-250)
o_pos_y = (250, b_mega_size[1]-250)
o_is_model = False

# branches
s_count = (0, 15)
s_gamma = (0.4, 2.4)
s_scale = (1.0, 2.1)
s_distance_min = 500
s_distance_max = 1400
s_blur = [1, 3, 5, 9, 15]

for img_number in range(out_img_count):
  t = time.time()
  filename = str(time.time())

  # background
  out = random_img(backgrounds_dir)[:,:,:3]

  out = cv2.resize(out, b_mega_size, cv2.INTER_AREA)

  rand_bkg_gamma = random.uniform(b_gamma[0], b_gamma[1])
  out = adjust_gamma(out, rand_bkg_gamma)
  if random.randint(0, 1) == 1:
    out = cv2.flip(out, 1)
    
  # luminare
  if not o_is_model:
    lum = random_img(img_dir)
    lum = random_size(lum, o_scale[0], o_scale[1])
    mask = lum[:,:,3]
    lum = lum[:,:,:3]
  else:
    lum = random_img(img_dir)[:,:,:3]
    lum = random_size(lum, o_scale[0], o_scale[1])
    mask = get_mask(lum, thr1=15, thr2=255)
  mask = cv2.medianBlur(mask, 9)

  lum = adjust_gamma(lum, rand_bkg_gamma)

  if not o_is_model :
    c = ((0, 0), (lum.shape[1], lum.shape[0]))
  else:
    c = get_contours(mask)
    lum = crop_by_countour(lum, c[0])
    mask = crop_by_countour(mask, c[0])

  o_x, o_y = random_position(o_pos_x, o_pos_y)
  out = combine_imgs(out, lum, mask, o_x, o_y)

  o_h, o_w = lum.shape[:2]
  c = ((np.clip(o_x - o_w // 2, 0, b_mega_size[0]),
        np.clip(o_y - o_h // 2, 0, b_mega_size[0])),
       (np.clip(o_x + o_w // 2, 0, b_mega_size[0]),
        np.clip(o_y + o_h // 2, 0, b_mega_size[0])))


  # branches
  for branches_count in range(random.randint(s_count[0], s_count[1])):
    s = random_img(branches_dir)

    s = adjust_gamma(s, rand_bkg_gamma)
    s = random_size(s, s_scale[0], s_scale[1])
    if random.randint(0, 1) == 1:
      s = cv2.flip(s, random.randint(-1, 1))
      
    mask = s[:,:,3]
    blur_p = s_blur[random.randint(0, len(s_blur) - 1)]
    s = cv2.GaussianBlur(s,(blur_p, blur_p), 0)
    mask = cv2.GaussianBlur(mask,(blur_p, blur_p), 0)

    angle = random.uniform(0, 6.28)
    l = random.randint(s_distance_min, s_distance_max)
    s_x = int(np.cos(angle) * l)
    s_y = int(np.sin(angle) * l)
    
    out = combine_imgs(out, s[:,:,:3], mask, o_x + s_x, o_y + s_y)

  out = cv2.resize(out, b_size, cv2.INTER_CUBIC)
  c = ((c[0][0] // 4, c[0][1] // 4), (c[1][0] // 4, c[1][1] // 4))

  if not True:
    cv2.rectangle(out, (c[0][0], c[0][1]), (c[1][0], c[1][1]), (255, 0, 0), 2)
    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  cv2.imwrite( '{}\\{}.{}'.format(save_dir, filename, ext), out)

  make_xml(filename, c, ext)
  print(time.time() - t)
