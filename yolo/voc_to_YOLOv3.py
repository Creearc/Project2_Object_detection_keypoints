import xml.etree.ElementTree as ET
from os import getcwd
import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-F", "--files", type=str, default='dataset/train/')
ap.add_argument("-N", "--name", type=str, default='LumDetector.txt')
args = vars(ap.parse_args())

dataset_train = args['files']
dataset_file = args['name']
classes_file = dataset_file[:-4]+'_classes.txt'


CLS = os.listdir(dataset_train)
classes =[dataset_train+CLASS for CLASS in CLS]
wd = getcwd()


def test(fullname):
    bb = ""
    in_file = open(fullname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for i, obj in enumerate(root.iter('object')):
        difficult = obj.find('difficult').text
        cls = fullname.split('/')[-2]# obj.find('name').text
        if cls not in CLS or int(difficult)==1:
            continue
        cls_id = CLS.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb += (" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        # we need this because I don't know overlapping or something like that
        if cls == 'Traffic_light':
            list_file = open(dataset_file, 'a')
            file_string = str(fullname)[:-4]+'.jpg'+bb+'\n'
            list_file.write(file_string)
            list_file.close()
            bb = ""

    if bb != "":
        list_file = open(args['name'], 'a')
        file_string = str(fullname)[:-4]+'.jpg'+bb+'\n'
        list_file.write(file_string)
        list_file.close()



for CLASS in classes:
    for filename in os.listdir(CLASS):
        if not filename.endswith('.xml'):
            continue
        fullname = os.getcwd()+'/'+CLASS+'/'+filename
        test(fullname)

for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS)+"\n"
    print(file_string)
    list_file.write(file_string)
    list_file.close()
