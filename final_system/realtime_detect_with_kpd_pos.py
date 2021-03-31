"""
python realtime_detect_with_kpd.py --shape-predictor kplum_gen_500.dat
python realtime_detect_with_kpd.py --shape-predictor kplum_gen_1000.dat
python realtime_detect_with_kpd.py --shape-predictor kplum_gen_1000_2.dat
python realtime_detect_with_kpd.py --shape-predictor kplum_gen_fix.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_fix2.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_rs.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_first_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_second_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_third_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_fourth_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_fifth_comb_test.dat

python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_sixth_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_seventh_comb_test.dat
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_eighth_comb_test.dat
"""
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

import multiprocessing
from multiprocessing import Pipe
import mss
import time

from imutils import face_utils
import argparse
import imutils
import dlib

LumConnectKeypointsPairs = [[0,1],[1,2], [2,3], [3,0], [0,4], [1,4], [2,4], [3,4]]
LumConnectKeypointsColors = [[200, 0, 0], [200, 200, 0], [200, 0, 200], [0, 100, 50], [100, 100, 100], [0, 200, 0]]

def connect_points_1(img, points, left, top):
    #img_h, img_w = img.shape[:2]

    lineThickness = 2
    cv2.line(img, (points[0][0]+left,points[0][1]+top), (points[1][0]+left,points[1][1]+top), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (points[1][0]+left,points[1][1]+top), (points[2][0]+left,points[2][1]+top), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (points[2][0]+left,points[2][1]+top), (points[3][0]+left,points[3][1]+top), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (points[3][0]+left,points[3][1]+top), (points[0][0]+left,points[0][1]+top), LumConnectKeypointsColors[3], lineThickness)

    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[0][0]+left,points[0][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[1][0]+left,points[1][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[2][0]+left,points[2][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[3][0]+left,points[3][1]+top), LumConnectKeypointsColors[4], lineThickness)
    return img

def connect_points_2(img, points, left, top):
    #img_h, img_w = img.shape[:2]

    lineThickness = 2
    cv2.line(img, (points[0][0]+left,points[0][1]+top), (points[2][0]+left,points[2][1]+top), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (points[1][0]+left,points[1][1]+top), (points[3][0]+left,points[3][1]+top), LumConnectKeypointsColors[1], lineThickness)

    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[0][0]+left,points[0][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[1][0]+left,points[1][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[2][0]+left,points[2][1]+top), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (points[3][0]+left,points[3][1]+top), LumConnectKeypointsColors[4], lineThickness)
    return img

def connect_points_3(img, points, left, top):
    from math import sqrt
    #img_h, img_w = img.shape[:2]
    
    lineThickness = 2
    cv2.line(img, (points[0][0]+left,points[0][1]+top), (points[2][0]+left,points[2][1]+top), LumConnectKeypointsColors[0], lineThickness)
    #cv2.line(img, (points[1][0]+left,points[1][1]+top), (points[3][0]+left,points[3][1]+top), LumConnectKeypointsColors[1], lineThickness)

    c = sqrt((points[1][0]-points[3][0])**2 + (points[1][1]-points[3][1])**2)

    a1 = points[0][0]+left-int(c/2)
    a2 = points[0][1]+top
    b1 = points[0][0]+left+int(c/2)
    b2 = points[0][1]+top
    c1 = points[2][0]+left-int(c/2)
    c2 = points[2][1]+top
    d1 = points[2][0]+left+int(c/2)
    d2 = points[2][1]+top

    cv2.line(img, (a1,a2),(b1,b2), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1,a2),(c1,c2), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1,b2),(d1,d2), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1,c2),(d1,d2), LumConnectKeypointsColors[3], lineThickness)

    cv2.line(img, (points[4][0]+left,points[4][1]+top), (a1,a2), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (b1,b2), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (c1,c2), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (points[4][0]+left,points[4][1]+top), (d1,d2), LumConnectKeypointsColors[4], lineThickness)
    return img

def connect_points_4(img, points, left, top):
    from math import sqrt    
    lineThickness = 2    

    c = sqrt((points[1][0]-points[3][0])**2 + (points[1][1]-points[3][1])**2)

    a1 = points[0][0]+left-int(c/2)
    a2 = points[0][1]+top
    b1 = points[0][0]+left+int(c/2)
    b2 = points[0][1]+top
    c1 = points[2][0]+left-int(c/2)
    c2 = points[2][1]+top
    d1 = points[2][0]+left+int(c/2)
    d2 = points[2][1]+top
    k = points[4][0]+left
    l = points[4][1]+top

    x1 = points[0][0]+left
    y1 = points[0][1]+top
    x2 = points[2][0]+left
    y2 = points[2][1]+top
    alpha = x2 - x1 #abs(x2 - x1)
    beta = y2 - y1 #abs(y2 - y1)

    y = int(((alpha**2/beta)*y1 + alpha*(k-x1) + beta*l)/((alpha**2/beta)+beta))
    x = int((alpha/beta)*(y - y1) + x1)

    cv2.line(img, (a1,a2),(b1,b2), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1,a2),(c1,c2), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1,b2),(d1,d2), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1,c2),(d1,d2), LumConnectKeypointsColors[3], lineThickness)

    cv2.line(img, (a1,a2),(d1,d2), LumConnectKeypointsColors[5], lineThickness)
    cv2.line(img, (b1,b2),(c1,c2), LumConnectKeypointsColors[5], lineThickness)

    p = x - k
    q = y - l

    cv2.line(img, (a1 - p,a2 - q),(b1 - p,b2 - q), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1 - p,a2 - q),(c1 - p,c2 - q), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1 - p,b2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1 - p,c2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[3], lineThickness)    

    cv2.line(img, (a1,a2),(a1 - p,a2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (c1,c2),(c1 - p,c2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (b1,b2),(b1 - p,b2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (d1,d2),(d1 - p,d2 - q), LumConnectKeypointsColors[4], lineThickness)    

    #cv2.line(img, (points[4][0]+left,points[4][1]+top), (x,y), LumConnectKeypointsColors[4], lineThickness) # высота
    return img

def connect_points_5(img, points, left, top):
    from math import sqrt
    lineThickness = 2                

    c = sqrt((points[1][0]-points[3][0])**2 + (points[1][1]-points[3][1])**2)

    a1 = points[0][0]+left-int(c/2)
    a2 = points[0][1]+top
    b1 = points[0][0]+left+int(c/2)
    b2 = points[0][1]+top
    c1 = points[2][0]+left-int(c/2)
    c2 = points[2][1]+top
    d1 = points[2][0]+left+int(c/2)
    d2 = points[2][1]+top
    k = points[4][0]+left
    l = points[4][1]+top

    x1 = points[0][0]+left
    y1 = points[0][1]+top
    x2 = points[2][0]+left
    y2 = points[2][1]+top
    alpha = x2 - x1 #abs(x2 - x1)
    beta = y2 - y1 #abs(y2 - y1)

    y = int(((alpha**2/beta)*y1 + alpha*(k-x1) + beta*l)/((alpha**2/beta)+beta))
    x = int((alpha/beta)*(y - y1) + x1)

    p = x - k
    q = y - l

    cv2.line(img, (a1 - p,a2 - q),(b1 - p,b2 - q), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1 - p,a2 - q),(c1 - p,c2 - q), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1 - p,b2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1 - p,c2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[3], lineThickness)    

    cv2.line(img, (a1 + p,a2 + q),(b1 + p,b2 + q), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1 + p,a2 + q),(c1 + p,c2 + q), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1 + p,c2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[3], lineThickness)

    cv2.line(img, (a1 + p,a2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[5], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(c1 + p,c2 + q), LumConnectKeypointsColors[5], lineThickness)

    cv2.line(img, (a1 + p,a2 + q),(a1 - p,a2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (c1 + p,c2 + q),(c1 - p,c2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(b1 - p,b2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (d1 + p,d2 + q),(d1 - p,d2 - q), LumConnectKeypointsColors[4], lineThickness)    

    #cv2.line(img, (points[4][0]+left,points[4][1]+top), (x,y), LumConnectKeypointsColors[4], lineThickness) # высота
    return img

def connect_points_6(img, points, left, top):
    from math import sqrt
    lineThickness = 2                

    #c = sqrt((points[1][0]-points[3][0])**2 + (points[1][1]-points[3][1])**2)
    cx = points[1][0]-points[3][0]
    cy = points[1][1]-points[3][1]

    a1 = points[0][0]+left-int(cx/2)
    a2 = points[0][1]+top-int(cy/2)
    b1 = points[0][0]+left+int(cx/2)
    b2 = points[0][1]+top+int(cy/2)
    c1 = points[2][0]+left-int(cx/2)
    c2 = points[2][1]+top-int(cy/2)
    d1 = points[2][0]+left+int(cx/2)
    d2 = points[2][1]+top+int(cy/2)
    k = points[4][0]+left
    l = points[4][1]+top

    x1 = points[0][0]+left
    y1 = points[0][1]+top
    x2 = points[2][0]+left
    y2 = points[2][1]+top
    alpha = x2 - x1
    beta = y2 - y1

    y = int(((alpha**2/beta)*y1 + alpha*(k-x1) + beta*l)/((alpha**2/beta)+beta))
    x = int((alpha/beta)*(y - y1) + x1)

    p = x - k
    q = y - l

    cv2.line(img, (a1 - p,a2 - q),(b1 - p,b2 - q), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1 - p,a2 - q),(c1 - p,c2 - q), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1 - p,b2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1 - p,c2 - q),(d1 - p,d2 - q), LumConnectKeypointsColors[3], lineThickness)    

    cv2.line(img, (a1 + p,a2 + q),(b1 + p,b2 + q), LumConnectKeypointsColors[0], lineThickness)
    cv2.line(img, (a1 + p,a2 + q),(c1 + p,c2 + q), LumConnectKeypointsColors[1], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[2], lineThickness)
    cv2.line(img, (c1 + p,c2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[3], lineThickness)

    cv2.line(img, (a1 + p,a2 + q),(d1 + p,d2 + q), LumConnectKeypointsColors[5], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(c1 + p,c2 + q), LumConnectKeypointsColors[5], lineThickness)

    cv2.line(img, (a1 + p,a2 + q),(a1 - p,a2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (c1 + p,c2 + q),(c1 - p,c2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (b1 + p,b2 + q),(b1 - p,b2 - q), LumConnectKeypointsColors[4], lineThickness)
    cv2.line(img, (d1 + p,d2 + q),(d1 - p,d2 - q), LumConnectKeypointsColors[4], lineThickness)    
    return img


# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0

start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 160, "left": 100, "width": 800, "height": 600}

#________
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

predictor = dlib.shape_predictor(args["shape_predictor"])
n = 0
#fls = os.listdir(img_path)
font = cv2.FONT_HERSHEY_SIMPLEX
#________

class YOLO(object):
    _defaults = {
        #"model_path": 'logs/2_1_trained_weights_final_t_3dmodel.h5',
        #"model_path": 'logs/9_3_trained_weights_final_t_combine_depth_median.h5',
        #"model_path": 'logs/20000_realsense_graphcut_trained_weights_final_tiny.h5',
        "model_path": 'logs/40400_trained_weights_final_t.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'LumDetector_classes.txt',
        "score" : 0.3,
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
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
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
            self.yolo_model.load_weights(self.model_path)
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

        np.random.shuffle(self.colors) 

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
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
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

            #________
            kp_crop = image[top:bottom+1, left:right+1]
            kp_gray = cv2.cvtColor(kp_crop, cv2.COLOR_BGR2GRAY)
            kp_rect = dlib.rectangle(0, 0, kp_crop.shape[1], kp_crop.shape[0])
            kp_shape = predictor(kp_gray, kp_rect)
            kp_shape = face_utils.shape_to_np(kp_shape)
            i_kp = -1
            #________
            # put object rectangle
            #cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)
            #________
            # put object key points with point number
            for (sX, sY) in kp_shape:
                #cv2.circle(image, (sX + left, sY + top), 3, (0, 0, 255), -1)
                cv2.putText(image, str(i_kp + 1), (sX + left, sY + top), font,0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(i_kp + 1), (sX + left, sY + top), font,0.8, (255, 255, 255), 1, cv2.LINE_AA)
                i_kp += 1
            #________
            # put 3D bounding box
            image = connect_points_6(image, kp_shape, left, top)

            '''
            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)
            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)
            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)
            '''
            
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


def GRABMSS_screen(p_input):
    while True:
        #Grab screen image
        img = np.array(sct.grab(monitor))

        # Put image from pipe
        p_input.send(img)
        
def SHOWMSS_screen(p_output):
    global fps, start_time
    yolo = YOLO()
    while True:
        img = p_output.recv()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        r_image, ObjectsList = yolo.detect_image(img)

        cv2.imshow("YOLO v3", r_image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    yolo.close_session()

        
if __name__=="__main__":
    p_output, p_input = Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(p_output,))

    # starting processes
    p1.start()
    p2.start()
