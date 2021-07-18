import torch.nn as nn


class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=1):
		super(ConvBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvNet(nn.Module):
	'''
	Network that composed by layers of ConvBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
