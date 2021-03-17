import os
import numpy as np
import cv2
import time


def make_xml(name, bbox, points, path='out'):
  f = open('{}{}.xml'.format(path, name), 'w')
  f.write('''<box top='{}' left='{}' width='{}' height='{}'>\n'''.format(bbox[1], bbox[0], bbox[2], bbox[3]))
  for i in range(len(points)):
    f.write('''<part name='{}' x='{}' y='{}'/>\n'''.format(i, points[i][0], points[i][1]))
  f.write('''</box>\n''')
  f.close()

font = cv2.FONT_HERSHEY_SIMPLEX 

img_dir = 'A:/Projects/Sortomat_2/dataset_alpha/bottles/PET_Blue/'
img_list = os.listdir(img_dir)
n = 0

bbox = None
points = []

save_path = 'annotated/'

while True:

  frame = cv2.imread('{}{}'.format(img_dir, img_list[n]), cv2.IMREAD_UNCHANGED)

  img = frame.copy()

  if bbox != None:
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,255,0), 1)

  for i in range(len(points)):
    cv2.circle(frame,(points[i][0], points[i][1]), 3, (0,0,255), -1)
    cv2.putText(frame, str(i + 1), (points[i][0], points[i][1]), font,
                0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(i + 1), (points[i][0], points[i][1]), font,
                0.8, (255, 255, 255), 1, cv2.LINE_AA) 
    
  cv2.imshow('', frame)
  
  key = cv2.waitKey()
  if key == ord('q') or key == 27:
    break
  elif key == ord('a'):
    if n > 0:
      n -= 1
  elif key == ord('d'):
    if n < len(img_list) - 1:
      n += 1
  elif key == ord('A'):
    n -= 50
  elif key == ord('D'):
    n += 50
  elif key == ord('b'):
    bbox = cv2.selectROI("", frame, fromCenter=False, showCrosshair=True)
  elif key == ord('p'):
    box = cv2.selectROI("", frame, fromCenter=True, showCrosshair=True)
    points.append((box[0] + box[2] // 2, box[1] + box[3] // 2))
  elif key == ord('c'):
    bbox = None
    points = []

  elif key == ord('S'):
    name = time.time()
    cv2.imwrite('{}{}.png'.format(save_path, name), img)
    make_xml(name, bbox, points, path=save_path)

cv2.destroyAllWindows()
