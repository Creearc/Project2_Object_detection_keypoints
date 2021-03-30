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

## Intel camera
In folder 
```
intel_realsense_camera/
```
you can find some programs which allows to capture data from intel realsense camera.
