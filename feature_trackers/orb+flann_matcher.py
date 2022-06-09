
import sys
sys.path.append(".")

import cv2
import numpy as np

import utils.feature_counter as feature_counter
import utils.fps_counter as fps_counter

video = cv2.VideoCapture("./data/Contesto_industriale1.mp4")

FEATURE_CAP = 1000
MIN_DISTANCE_THRESHOLD = 80

color = np.random.randint(0, 255, (FEATURE_CAP, 3))

featureCounter = feature_counter.FeatureCounter()
fpsCounter = fps_counter.FPSCounter()

last_frame_features = None
last_frame_kp = None
last_frame_image = None
	
orb = cv2.ORB_create(nfeatures=FEATURE_CAP)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
				   table_number = 6, # 12
				   key_size = 12,     # 20
				   multi_probe_level = 1) #2
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)

while video.isOpened():

	ret, frame = video.read()
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	keypoints, descriptors = orb.detectAndCompute(gray_image, None)
	
	if last_frame_features is not None and descriptors is not None: 
		matches = matcher.knnMatch(last_frame_features, descriptors, k=2)
		matchesMask = [[0,0] for _ in range(len(matches))]
		
		for i, match_pair in enumerate(matches):
			try: 
				descriptor1, descriptor2 = match_pair
			except: 
				continue

			if descriptor1.distance < 0.5 * descriptor2.distance:
				matchesMask[i]=[1,0]

		draw_params = dict(
			matchColor = (0,255,0), 
			singlePointColor = (255,0,0),
			matchesMask = matchesMask,
			flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
		)

		img = cv2.drawMatchesKnn(last_frame_image, last_frame_kp, frame, keypoints, matches, None, **draw_params)
	
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
