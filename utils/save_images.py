'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def save_images(saved_all, epoch, folder, resize_to=None):
	original_images, watermarked_images, noised_images = saved_all

	images = original_images[:original_images.shape[0], :, :, :].cpu()
	watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

	# scale values to range [0, 1] from original range of [-1, 1]
	images = (images + 1) / 2
	watermarked_images = (watermarked_images + 1) / 2
	noised_images = (noised_images + 1) / 2

	# resize noised_images
	resize = nn.UpsamplingNearest2d(size=(images.shape[2], images.shape[3]))
	noised_images = resize(noised_images)

	if resize_to is not None:
		images = F.interpolate(images, size=resize_to)
		watermarked_images = F.interpolate(watermarked_images, size=resize_to)

	diff_images = (watermarked_images - images + 1) / 2

	# transform to rgb
	diff_images_linear = diff_images.clone()
	R = diff_images_linear[:, 0, :, :]
	G = diff_images_linear[:, 1, :, :]
	B = diff_images_linear[:, 2, :, :]
	diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
	diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

	# maximize diff in every image
	for id in range(diff_images_linear.shape[0]):
		diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
				diff_images_linear[id].max() - diff_images_linear[id].min())

	stacked_images = torch.cat(
		[images.unsqueeze(0), watermarked_images.unsqueeze(0), noised_images.unsqueeze(0), diff_images.unsqueeze(0),
		 diff_images_linear.unsqueeze(0)], dim=0)
	shape = stacked_images.shape
	stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
	stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

	saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
	saved_image.save(filename)


def get_random_images(images, encoded_images, noised_images):
	selected_id = np.random.randint(1, images.shape[0]) if images.shape[0] > 1 else 1
	image = images.cpu()[selected_id - 1:selected_id, :, :, :]
	encoded_image = encoded_images.cpu()[selected_id - 1:selected_id, :, :, :]
	noised_image = noised_images.cpu()[selected_id - 1:selected_id, :, :, :]
	return [image, encoded_image, noised_image]


def concatenate_images(saved_all, images, encoded_images, noised_images):
	saved = get_random_images(images, encoded_images, noised_images)
	if saved_all[2].shape[2] != saved[2].shape[2]:
		return saved_all
	saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
	saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
	saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
	return saved_all
