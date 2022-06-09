
import sys
sys.path.append(".")

import cv2
import numpy as np

import utils.feature_counter as feature_counter
import utils.fps_counter as fps_counter

FEATURE_CAP = 1000
MIN_DISTANCE_THRESHOLD = 80

video = cv2.VideoCapture("./data/Contesto_industriale1.mp4")

color = np.random.randint(0, 255, (FEATURE_CAP, 3))

featureCounter = feature_counter.FeatureCounter()
fpsCounter = fps_counter.FPSCounter()

last_frame_features = None
last_frame_kp = None
last_frame_image = None
	
orb = cv2.ORB_create(nfeatures=FEATURE_CAP)
matcher = cv2.BFMatcher(cv2.NORM_L2)

while video.isOpened():

	ret, frame = video.read()
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	keypoints, descriptors = orb.detectAndCompute(gray_image, None)
	
	if last_frame_features is not None and descriptors is not None: 
		matches = matcher.match(last_frame_features, descriptors)		
		good_matches = list(filter(lambda x: x.distance < MIN_DISTANCE_THRESHOLD, matches))

		draw_params = dict(
			matchColor = (0,255,0), 
			singlePointColor = (255,0,0),
			flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
		)

		img = cv2.drawMatches(last_frame_image, last_frame_kp, frame, keypoints, good_matches, None, **draw_params)	
		cv2.imshow('video', img)	

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	last_frame_features = descriptors
	last_frame_kp = keypoints
	last_frame_image = frame

	featureCounter.update(len(descriptors))
	fpsCounter.update()


featureCounter.print_feature_per_frame()
fpsCounter.print_avg_fps()

video.release()
cv2.destroyAllWindows()
