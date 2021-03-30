import numpy as np
import cv2
import time
import os

def add_annotation(path, annotation):
  f.write('''  <image file='{}'>\n'''.format(path))
  f.write('''{}'''.format(annotation))
  f.write('''  </image>\n''')

f = open('{}.xml'.format('final_annotation'), 'w')
f.write('''<dataset>\n''')
f.write('''  <images>\n''')

imgext = 'jpg'

xml_dir = 'img'
for file in os.listdir(xml_dir):
  xmlname = file
  print(xmlname)
  if xmlname[-3:] == 'xml':
    with open('{}/{}'.format(xml_dir,xmlname), 'r') as xml_file:
      text = xml_file.read()#.replace('<box/>', '</box>')
      print(text)
      path = os.path.abspath('{}/{}'.format(xml_dir,xmlname))
      path = path.replace('\\','/').replace('xml',imgext)
      #path = path.replace('\\','/').replace('xml','jpg')
      print(path)
      add_annotation(path, text)
f.write('''  </images>\n''')
f.write('''</dataset>''')
f.close()