
import cv2
import numpy as np


# kalman = cv2.KalmanFilter(4,2)
# kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
# kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
# kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1


# def process_keypoint(key : cv2.KeyPoint): 
# 	current_mes = np.array([[np.float32(key.pt[0])],[np.float32(key.pt[1])]])

	# kalman.correct(current_mes)
	# current_pre = kalman.predict()

	# if last_mes is not None: 
	# 	lmx, lmy = last_mes[0].astype(int),last_mes[1].astype(int)
	# 	lpx, lpy = last_pre[0].astype(int),last_pre[1].astype(int)

	# 	cmx, cmy = current_mes[0].astype(int),current_mes[1].astype(int)    
	# 	cpx, cpy = current_pre[0].astype(int),current_pre[1].astype(int)   
		
	# 	cv2.line(frame, (lmx[0],lmy[0]),(cmx[0],cmy[0]),(0,255,0))
	# 	cv2.line(frame, (lpx[0],lpy[0]),(cpx[0],cpy[0]),(0,0,255))




video = cv2.VideoCapture("Contesto_industriale1.mp4")

last_mes = None
last_pre = None

frame_index = 0
while video.isOpened():

	ret, frame = video.read()

	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
	# sift = cv2.xfeatures2d.SIFT_create()
	# kp, des = sift.detectAndCompute(gray_image, None)

	orb = cv2.ORB_create(nfeatures=1000)
	kp, des = orb.detectAndCompute(gray_image, None)

	first_key = kp[0]
	current_mes = np.array([[np.float32(first_key.pt[0])],[np.float32(first_key.pt[1])]])

	# kalman.correct(current_mes)
	# current_pre = kalman.predict()

	# if last_mes is not None: 
	# 	lmx, lmy = last_mes[0].astype(int),last_mes[1].astype(int)
	# 	lpx, lpy = last_pre[0].astype(int),last_pre[1].astype(int)

	# 	cmx, cmy = current_mes[0].astype(int),current_mes[1].astype(int)    
	# 	cpx, cpy = current_pre[0].astype(int),current_pre[1].astype(int)   
		
	# 	cv2.line(frame, (lmx[0],lmy[0]),(cmx[0],cmy[0]),(0,255,0))
	# 	cv2.line(frame, (lpx[0],lpy[0]),(cpx[0],cpy[0]),(0,0,255))

	kp_image = cv2.drawKeypoints(frame, kp, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
	cv2.imshow('video', kp_image)	

	# cv2.imshow('video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	# last_mes = current_mes
	# last_pre = current_pre

	frame_index += 1

video.release()
cv2.destroyAllWindows()
