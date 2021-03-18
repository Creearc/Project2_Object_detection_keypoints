import os
import cv2
import numpy as np
import time

def test(img_name, threshold=100, l=10):
  img = cv2.imread('{}_i.png'.format(img_name))
  depth = cv2.imread('{}_c.png'.format(img_name))
  depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(depth, threshold, 255, cv2.THRESH_BINARY_INV)
  ret, mask2 = cv2.threshold(depth, threshold - l, 255, cv2.THRESH_BINARY)
  #mask2 = cv2.bitwise_and(mask2)
  return img, cv2.bitwise_and(mask, mask2)


if __name__ == '__main__':
  font = cv2.FONT_HERSHEY_SIMPLEX
  color = (255)
  fontScale = 1
  thickness = 2
  org = (50, 50)

  img_dir = 'people/'
  save_dir = 'samples/'
  
  thr = 125
  l = 30
  away = False
  for filename in os.listdir(img_dir):
    if away:
      break
    if filename.split('_')[1][0] == 'i':
      filename = filename.split('_')[0]
    else:
      continue

    img = cv2.imread('{}{}_i.png'.format(img_dir, filename))
    depth = cv2.imread('{}{}_c.png'.format(img_dir, filename))
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    h, w, ch = img.shape
    box = (0, 0, w, h)
    out = np.zeros((h, w, 4),np.uint8)
    while True:
      ret, mask = cv2.threshold(depth, thr, 255, cv2.THRESH_BINARY_INV)
      ret, mask2 = cv2.threshold(depth, thr - l, 255, cv2.THRESH_BINARY)
      mask = cv2.bitwise_and(mask, mask2)
      
      out[:,:,:3] = cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
      out[:,:,3] = mask
      #out = cv2.bitwise_and(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
      frame = out.copy()
      cv2.putText(frame, 'thr:{} len:{}'.format(thr, l), org, font, fontScale, 255, thickness+3, cv2.LINE_AA)
      cv2.putText(frame, 'thr:{} len:{}'.format(thr, l), org, font, fontScale, 0, thickness, cv2.LINE_AA)
      cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
      cv2.imshow('test', frame)
      cv2.imshow('orig', img)
      #cv2.imshow('test2', img2)
      key = cv2.waitKey(1)
      if key == ord("a"):
        if thr - l < 0:
          thr = l
        else:
          thr -= 1
      elif key == ord("d"):
        if thr > 255:
          thr = 255
        else:
          thr += 1
      elif key == ord("1") and l > 0:
        l -= 1
      elif key == ord("2") and l < 255:
        l += 1
      elif key == ord("q") or key == 27:
        away = True
        break
      elif key == ord("s"):
        box = cv2.selectROI('test', out, fromCenter=False,showCrosshair=True)
        print(box)
      elif key == ord("w"):
        cv2.imwrite( '{}\\{}.png'.format(save_dir, time.time()), out[box[1] : box[1] + box[3], box[0] : box[0] + box[2]])
        break
      elif key == ord("e"):
        break
        
    cv2.destroyAllWindows()
