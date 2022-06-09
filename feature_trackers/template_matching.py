
import sys
sys.path.append(".")

import cv2
import numpy as np
import utils.feature_counter as feature_counter
import utils.fps_counter as fps_counter

THRESHOLD = 0.47

template_img = cv2.imread('./data/template3.png', 0)

w, h = template_img.shape[::-1]
w, h = w*2, h*2 
video = cv2.VideoCapture("./data/Contesto_industriale1.mp4")

featureCounter = feature_counter.FeatureCounter()
fpsCounter = fps_counter.FPSCounter()

while video.isOpened():

	ret, frame = video.read()
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
	res = cv2.matchTemplate(gray_image, template_img, cv2.TM_CCOEFF_NORMED)

	loc = np.where(res >= THRESHOLD)
	for pt in zip(*loc[::-1]):
		cv2.rectangle(frame, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), 255, 5)

	cv2.imshow('matches', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	featureCounter.update(len(res))
	fpsCounter.update()


featureCounter.print_feature_per_frame()
fpsCounter.print_avg_fps()

video.release()
cv2.destroyAllWindows()





