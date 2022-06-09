
import sys
sys.path.append(".")

import cv2
import numpy as np

import utils.feature_counter as feature_counter
import utils.fps_counter as fps_counter

video = cv2.VideoCapture("./data/Contesto_industriale1.mp4")

MIN_NUMBER_OF_ACTIVE_FEATURES = 4

featureCounter = feature_counter.FeatureCounter()
fpsCounter = fps_counter.FPSCounter()

feature_params = dict( 
	maxCorners = 100, 
	qualityLevel = 0.3, 
	minDistance = 7, 
	blockSize = 7 
)

lk_params = dict( 
	winSize  = (15, 15), 
	maxLevel = 2, 
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

color = np.random.randint(0, 255, (100, 3))

previous_features = []

while video.isOpened():

	ret, frame = video.read()
	if not ret: 
		break

	if len(previous_features) < MIN_NUMBER_OF_ACTIVE_FEATURES: 
		old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		previous_features = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
	current_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_image, previous_features, None, **lk_params)

	if current_features is not None:
		good_current_features = current_features[status==1]
		good_previous_features = previous_features[status==1]

	for i, current_feat in enumerate(good_current_features):
		cx, cy = current_feat.ravel()
		frame = cv2.circle(frame, (int(cx), int(cy)), 15, color[i].tolist(), 12)	

	cv2.imshow('video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	old_gray = gray_image.copy()
	previous_features = good_current_features.reshape(-1, 1, 2)

	featureCounter.update(len(good_current_features))
	fpsCounter.update()


featureCounter.print_feature_per_frame()
fpsCounter.print_avg_fps()

video.release()
cv2.destroyAllWindows()