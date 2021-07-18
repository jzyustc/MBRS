import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BasicBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
					  stride=drop_rate, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class BottleneckBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BottleneckBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class SENet(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class SENet_decoder(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
