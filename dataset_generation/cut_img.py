import os
import cv2
import numpy as np
import time


if __name__ == '__main__':
  img_dir = 'datasets_people/people_2/'
  save_dir = 'datasets_people/samples/'
  
  away = False
  for filename in os.listdir(img_dir):
    print(filename)
    if away:
      break

    img = cv2.imread('{}{}'.format(img_dir, filename), cv2.IMREAD_UNCHANGED)

    h, w, ch = img.shape
    box = (0, 0, w, h)

    while True:
      frame = img.copy()

      cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
      cv2.imshow('test', frame)
      key = cv2.waitKey(1)

      if key == ord("q") or key == 27:
        away = True
        break
      elif key == ord("b"):
        box = cv2.selectROI('test', frame, fromCenter=False,showCrosshair=True)
        print(box)
      elif key == ord("w"):
        cv2.imwrite( '{}/{}.png'.format(save_dir, time.time()), img[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :])
        break
      elif key == ord("e"):
        break
        
    cv2.destroyAllWindows()
