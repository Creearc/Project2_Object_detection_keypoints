# Project2_Object_detection_keypoints

## Dataset generation
Folder
```
dataset_generation/
```
All programs to annotate and generate data are placed in this folder.

## Yolo
Folder
```
yolo/
```

To train YOLO you should generate files with annotation. Use this program:
```
voc_to_YOLOv3.py
```

If you have one class of object you can use this pgrogam to train yolo model:
```
train.py
```

To train model whith more than one class use this program:
```
train_combine_2.py
```
## Dlib
Folder
```
dlib/
```
All programs to train dlib shape predictor for kepoints detection (required images annotated keypoints).

## Evaluation
Folder
```
evaluation/
```
All programs to evaluate yolo object detection models using Intersection ovaer Union and dlib shape predictor.

## Final system 
Folder
```
final_system/
```
All programs to run final system which detects object (luminaire), finds it's keypoints and draws the 3D bounding box. 
To run final system, use:
```
python realtime_detect_with_kpd_pos.py --shape-predictor kplum_gen_ninth_comb_test.dat
```
but it requires trained YOLO object detection model (```.h5```) in folder ```logs/``` and trained dlib shape predictor (```.dat```)

## Intel camera
In folder 
```
intel_realsense_camera/
```
you can find some programs which allows to capture data from intel realsense camera.
