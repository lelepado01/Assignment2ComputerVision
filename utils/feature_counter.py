
class FeatureCounter: 

	feats = 0
	frames = 0

	def update(self, feat_num, frame_num = 1): 
		self.feats += feat_num
		self.frames += frame_num

	def print_feature_per_frame(self): 
		print("Features/Frames: " + str(round(self.feats / self.frames, 2)))
