"""
python test_keypoints_images.py --shape-predictor kplum.dat
python test_keypoints_images.py --shape-predictor kplum_gen.dat
python test_keypoints_images.py --shape-predictor kplum_gen_500.dat
python test_keypoints_images.py --shape-predictor kplum_gen_fix2.dat

python test_keypoints_images.py --shape-predictor kplum_gen_third_comb_test.dat
"""

# import the necessary packages
#from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# trained shape predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
n = 0
img_path = 'Validation_photos_KP_cropped/'

fls = os.listdir(img_path)
font = cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
while True:
	if not (img_path + '/' +fls[n]).endswith('.jpg'):
		n += 1
	else:
		frame = cv2.imread(img_path + '/' +fls[n])
		frame = imutils.resize(frame, width=200)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
		# detect faces in the grayscale frame
		rect = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])


		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		i = -1
		for (sX, sY) in shape:
			cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
			cv2.putText(frame, str(i + 1), (sX, sY), font,0.8, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(frame, str(i + 1), (sX, sY), font,0.8, (255, 255, 255), 1, cv2.LINE_AA)
			i += 1

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop

		if key == ord('q') or key == 27:
			break
		elif key == ord('a'):
			if n > 0:
				n -= 1
		elif key == ord('d'):
			n += 1
		elif key == ord("q"):
			break
		      
	# do a bit of cleanup
cv2.destroyAllWindows()