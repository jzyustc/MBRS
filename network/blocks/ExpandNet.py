import torch.nn as nn


class ConvTBNRelu(nn.Module):
	"""
	A sequence of TConvolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=2):
		super(ConvTBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ExpandNet(nn.Module):
	'''
	Network that composed by layers of ConvTBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ExpandNet, self).__init__()

		layers = [ConvTBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvTBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
