import torch
import torch.nn as nn
import numpy as np

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Crop, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		mask = torch.zeros_like(image)
		mask[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask

class Cropout(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Cropout, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output

class Dropout(nn.Module):

	def __init__(self, prob):
		super(Dropout, self).__init__()
		self.prob = prob

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn > self.prob * 1., cover_image, image)
		return output
