"""
python YOLO_IoU_argparse.py --m=1 --exp=1 --csv=Validation_photos_IoU_labels.csv
"""
import argparse
import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
import time
import pandas as pd


parser = argparse.ArgumentParser(description='Evaluation of object detection model (YOLOv3-tiny) using Intersection over Union')
parser.add_argument('--model', type=int, default=1)
parser.add_argument('--experiment', type=int, default=1)
parser.add_argument('--csv', type=str, default="Validation_photos_IoU_labels.csv")
args = parser.parse_args()

m = args.model
exp = args.experiment

class YOLO(object):
    _defaults = {
        "model_path": 'logs/paper_models/{0}/{0}_{1}_trained_weights_final_t.h5'.format(str(m),str(exp)),
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'LumDetector_classes.txt',
        "score" : 0.5,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList

def iou(y_true, y_pred):
    # order is [x_left, y_top, x_right, y_bottom]
    intersection_xmin = max(y_true[0], y_pred[0])
    intersection_ymin = max(y_true[1], y_pred[1])
    intersection_xmax = min(y_true[2], y_pred[2])
    intersection_ymax = min(y_true[3], y_pred[3])
    area_intersection = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    area_y = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
    area_yhat = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])
    area_union = area_y + area_yhat - area_intersection

    iou = area_intersection/area_union
    iou = round(iou, 3)
    return iou
    
if __name__=="__main__":
    yolo = YOLO()

    csvname = args.csv
    trueList = pd.read_csv(csvname, delimiter=',')

    result = {'filename':[], 'XminT':[], 'YminT':[], 'XmaxT':[], 'YmaxT':[], 'Xmin':[], 'Ymin':[], 'Xmax':[], 'Ymax':[], 'IoU':[], 'time, sec': []}
    resultList = pd.DataFrame(result)

    i = 0
    img_dir = 'D:/LumRec/Validation_photos_KP/'
    sTime = time.time()
    print("IoU_{0}_{1}".format(str(m),str(exp)))

    for file in os.listdir(img_dir):
        startTime = time.time()

        imgname = file
        print(imgname)
        print(img_dir + imgname)

        image = cv2.imread(img_dir + imgname)
        r_image, ObjectsList = yolo.detect_img(img_dir + imgname)

        for index, row in trueList.iterrows():
            if row['filename'] == imgname:
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        y_true = [xmin, ymin, xmax, ymax]
        print(y_true)
        cv2.rectangle(r_image, (y_true[0], y_true[1]), (y_true[2], y_true[3]), (0, 255, 0), 2)      


        if not len(ObjectsList)==0:
            y_pred = [ObjectsList[0][1],ObjectsList[0][0],ObjectsList[0][3],ObjectsList[0][2]]
            print(y_pred)

            img_iou = iou(y_true, y_pred)
            if img_iou < 0:
                img_iou = -img_iou

            print('iou =', img_iou)
            recTime = time.time() - startTime
            print(recTime)
            print ('  ')

            new_row = pd.DataFrame({'filename': imgname, 
                                        'XminT': xmin, 'YminT': ymin, 'XmaxT': xmax, 
                                        'YmaxT': ymax, 'Xmin': ObjectsList[0][1], 'Ymin': ObjectsList[0][0], 
                                        'Xmax': ObjectsList[0][3], 'Ymax': ObjectsList[0][2], 
                                        'IoU': img_iou, 'time, sec': recTime}, index = [i])

            resultList = pd.concat([resultList, new_row]).reset_index(drop = True)
            i = i + 1
        else:
            y_pred = [0,0,0,0]
            img_iou = 0
            print('iou =', img_iou)
            #cv2.imwrite( 'test_IoU_saved_images/{}.jpg'.format(imgname), r_image)
            recTime = time.time() - startTime
            print(recTime)
            print ('  ')

            new_row = pd.DataFrame({'filename': imgname, 
                                        'XminT': xmin, 'YminT': ymin, 'XmaxT': xmax, 
                                        'YmaxT': ymax, 'Xmin': y_pred[0], 'Ymin': y_pred[0], 
                                        'Xmax': y_pred[0], 'Ymax': y_pred[0], 
                                        'IoU': img_iou, 'time, sec': recTime}, index = [i])

            resultList = pd.concat([resultList, new_row]).reset_index(drop = True)
            i = i + 1

    print(resultList)
    print("logs/paper_models/{0}/IoU_{0}_{1}.xlsx".format(str(m),str(exp)))
    resultList.to_excel("logs/paper_models/{0}/IoU_{0}_{1}.xlsx".format(str(m), str(exp)))#combine_depth.xlsx")

    print(time.time()-sTime)
    yolo.close_session()