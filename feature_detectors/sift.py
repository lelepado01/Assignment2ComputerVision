
import sys
sys.path.append( '.' )

import cv2

import utils.feature_counter as feature_counter
import utils.fps_counter as fps_counter

video = cv2.VideoCapture("./data/Contesto_industriale1.mp4")

featureCounter = feature_counter.FeatureCounter()
fpsCounter = fps_counter.FPSCounter()

sift = cv2.xfeatures2d.SIFT_create()

while video.isOpened():

	ret, frame = video.read()

	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
	keypoints, descriptors = sift.detectAndCompute(gray_image, None)

	frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

	cv2.imshow('video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	featureCounter.update(len(keypoints))
	fpsCounter.update()


featureCounter.print_feature_per_frame()
fpsCounter.print_avg_fps()

video.release()
cv2.destroyAllWindows()
