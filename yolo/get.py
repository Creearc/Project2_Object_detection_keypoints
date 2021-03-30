# -*- coding: cp1251 -*-
from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

ftp = FTP()
HOST = '192.168.68.128'
#HOST = '192.168.137.164'
PORT = 21

ftp.connect(HOST, PORT)

print(ftp.login(user='alexandr', passwd='9'))


ftp.cwd('/home/alexandr/YOLO/YOLOv3-custom-training')

fl = 'out.zip'
out = '{}\{}'.format(PATH, fl)

with open(out, 'wb') as f:
    ftp.retrbinary('RETR ' + fl, f.write)

