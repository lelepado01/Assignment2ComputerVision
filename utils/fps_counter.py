
import time

class FPSCounter:  

	start_time = time.time()
	current_time = time.time() 

	total_frames = 0

	def update(self):
		self.total_frames += 1
		self.current_time = time.time()

	def print_fps(self): 
		print("FPS: " + str(round(1 / (time.time() - self.current_time), 2)))

	def print_avg_fps(self): 
		print("AVG FPS: " + str(round(self.total_frames / (time.time() - self.start_time), 2)))
