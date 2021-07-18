import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data


class Dataloader:

	def __init__(self, batch_size, path, H=256, W=256):
		self.batch_size = batch_size
		self.H = H
		self.W = W
		self.train_path = path + "train/"
		self.val_path = path + "validation/"
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def resize(self, image, W, H):
		min_rate = min(image.shape[0] / H, image.shape[1] / W)
		size = (int(image.shape[1] / min_rate), int(image.shape[0] / min_rate))
		image = Image.fromarray(np.uint8(image))
		image = image.resize(size, Image.ANTIALIAS)
		return np.array(image, dtype=np.uint8)

	def padding(self, image):

		pad_value = image.max()
		# pad the width if needed
		if image.shape[0] < self.W:
			pad_left = (self.W - image.shape[0]) // 2
			pad_right = (self.W - image.shape[0]) - pad_left
			image = np.pad(image, ((pad_left, pad_right), (0, 0), (0, 0)), 'constant', constant_values=pad_value)
		# pad the height if needed
		if image.shape[1] < self.H:
			pad_top = (self.H - image.shape[1]) // 2
			pad_bottom = (self.H - image.shape[1]) - pad_top
			image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (0, 0)), 'constant', constant_values=pad_value)

		return image

	def transform_image(self, image):

		# ignore
		if image.shape[0] < self.W / 2 and image.shape[1] < self.H / 2:
			return None
		if image.shape[0] < image.shape[1] / 2 or image.shape[1] < image.shape[0] / 2:
			return None

		# gray
		if len(image.shape) == 2:
			image = image[:, :, np.newaxis].repeat(3, 2)



		# Padding / Resize
		image = self.resize(image, self.W * 1.1, self.H * 1.1)
		image = self.padding(image)

		# Crop
		w_pos = random.randint(0, image.shape[0] - self.W)
		h_pos = random.randint(0, image.shape[1] - self.H)
		image = image[w_pos:w_pos + self.W, h_pos:h_pos + self.H, :]

		# ToTensor and Normalize
		image = self.transform(image)

		return image

	def load(self, path):
		data = []
		id = 0
		for image_name in os.listdir(path):
			# get transformed image & mask
			image = Image.open(path + image_name)
			image = np.array(image, dtype=np.float32)

			transformed = self.transform_image(image)
			id += 1

			try:

				if transformed is not None:
					data.append(transformed)
			except:
				print("ERROR in image : " + path + image_name)

		return data

	def load_train_data(self):
		train_data = self.load(self.train_path)
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
												   pin_memory=True)
		return train_loader

	def load_val_data(self):
		val_data = self.load(self.val_path)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
												 pin_memory=True)
		return val_loader
