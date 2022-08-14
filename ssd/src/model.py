# coding=utf-8
import os
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from src.resnet import wide_resnet50_2, wide_resnet50_3, resnext101_64x4d
from src.resnet import resnet50 as c_resnet50


class Base(nn.Module):
	def __init__(self):
		super().__init__()

	def init_weights(self):
		layers = [*self.additional_blocks, *self.loc, *self.conf]
		if hasattr(self, 'smooth'):
			layers.append(self.smooth)
		for layer in layers:
			for param in layer.parameters():
				if param.dim() > 1:
					nn.init.xavier_uniform_(param)

	def bbox_view(self, src, loc, conf):
		ret = []
		for s, l, c in zip(src, loc, conf):
			ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

		locs, confs = list(zip(*ret))
		locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
		return locs, confs


class ResNet(nn.Module):
	def __init__(self):
		super().__init__()
		backbone = resnet50(pretrained=True)
		self.out_channels = [1024, 512, 512, 256, 256, 256]
		self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

		conv4_block1 = self.feature_extractor[-1][0]
		conv4_block1.conv1.stride = (1, 1)
		conv4_block1.conv2.stride = (1, 1)
		conv4_block1.downsample[0].stride = (1, 1)

	def forward(self, x):
		x = self.feature_extractor(x)
		return x


class SSD(Base):
	def __init__(self, backbone=ResNet(), num_classes=81):
		super().__init__()

		self.feature_extractor = backbone
		self.num_classes = num_classes
		self._build_additional_features(self.feature_extractor.out_channels)
		self.num_defaults = [4, 6, 6, 6, 4, 4]
		self.loc = []
		self.conf = []

		for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
			self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
			self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

		self.loc = nn.ModuleList(self.loc)
		self.conf = nn.ModuleList(self.conf)
		self.init_weights()

	def _build_additional_features(self, input_size):
		self.additional_blocks = []
		for i, (input_size, output_size, channels) in enumerate(
				zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
			if i < 3:
				layer = nn.Sequential(
					nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
					nn.BatchNorm2d(channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
					nn.BatchNorm2d(output_size),
					nn.ReLU(inplace=True),
				)
			else:
				layer = nn.Sequential(
					nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
					nn.BatchNorm2d(channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
					nn.BatchNorm2d(output_size),
					nn.ReLU(inplace=True),
				)

			self.additional_blocks.append(layer)

		self.additional_blocks = nn.ModuleList(self.additional_blocks)


	def forward(self, x):
		x = self.feature_extractor(x)
		detection_feed = [x]
		for l in self.additional_blocks:
			x = l(x)
			detection_feed.append(x)
		locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
		
		return [locs,], [confs,]


class ResNetv2(nn.Module):
	def __init__(self, split_factor=1, pretrained_dir='None', arch='wide_resnet50_2'):
		super().__init__()
		self.split_factor = split_factor
		self.pretrained_dir = pretrained_dir
		models = []
		model_kwargs = {'num_classes': 1000,
						'dataset': 'imagenet',
						'split_factor': self.split_factor
						}

		print('INFO: [model] build with {} and split_factor {}'.format(arch, split_factor))
		for i in range(self.split_factor):
			if arch == 'wide_resnet50_2':
				backbone = wide_resnet50_2(pretrained=False, **model_kwargs)
			elif arch == 'wide_resnet50_3':
				backbone = wide_resnet50_3(pretrained=False, **model_kwargs)
			elif arch == 'resnext101_64x4d':
				backbone = resnext101_64x4d(pretrained=False, **model_kwargs)
			elif arch == 'resnet50':
				backbone = c_resnet50(pretrained=False, **model_kwargs)
			else:
				raise NotImplementedError

			feature_extractor = nn.Sequential(*list(backbone.children())[:4])
			# print(feature_extractor)
			# set the stride of the 3rd layer to 1
			conv4_block1 = feature_extractor[-1][0]
			conv4_block1.conv1.stride = (1, 1)
			conv4_block1.conv2.stride = (1, 1)
			# conv4_block1.downsample[0].stride = (1, 1)
			conv4_block1.downsample[0] = nn.Identity()
			models.append(feature_extractor)

		self.backbone = nn.ModuleList(models)
		self.out_channels = [1024, 512, 512, 256, 256, 256]
		if os.path.isfile(self.pretrained_dir):
			self.load_pretrain_weights()

	def forward(self, x):
		out_l = []
		for i in range(self.split_factor):
			out_l.append(self.backbone[i](x))
		
		return out_l

	def load_pretrain_weights(self):
		sd = torch.load(self.pretrained_dir, map_location='cpu')['state_dict']
		sd = dict([(k[14:].replace('.layer', '.'), v) for k, v in sd.items()])
		# print(sd.keys())
		# print(self.backbone.state_dict().keys())
		miss, unexpect = self.backbone.load_state_dict(sd, strict=False)
		print("missing_keys {}, unexpected_keys {}".format(miss, unexpect))

		print("\nLoaded base model.\n")


class SSDv2(Base):
	def __init__(self, split_factor=1, pretrained_dir='None', arch='wide_resnet50_2', num_classes=81, res_blocks=None):
		super().__init__()
		print("INFO:PyTorch: Build model with split_factor {}".format(split_factor))
	
		self.split_factor = split_factor
		self.feature_extractor = ResNetv2(split_factor=split_factor, pretrained_dir=pretrained_dir, arch=arch)
		self.num_classes = num_classes
		self._build_additional_features(self.feature_extractor.out_channels)
		self.num_defaults = [4, 6, 6, 6, 4, 4]
		self.loc = []
		self.conf = []

		for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
			self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
			self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

		self.loc = nn.ModuleList(self.loc)
		self.conf = nn.ModuleList(self.conf)
		self.init_weights()

	def _build_additional_features(self, input_size):
		self.additional_blocks = []
		for i, (input_size, output_size, channels) in enumerate(
				zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
			if i < 3:
				layer = nn.Sequential(
					nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
					nn.BatchNorm2d(channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
					nn.BatchNorm2d(output_size),
					nn.ReLU(inplace=True),
				)
			else:
				layer = nn.Sequential(
					nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
					nn.BatchNorm2d(channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
					nn.BatchNorm2d(output_size),
					nn.ReLU(inplace=True),
				)

			self.additional_blocks.append(layer)

		self.additional_blocks = nn.ModuleList(self.additional_blocks)


	def forward(self, x):
		locs_l, confs_l = [], []
		xs = self.feature_extractor(x)

		# plan 1: use the mean feature as input
		# x = torch.mean(torch.stack(xs), dim=0)
		# detection_feed = [x]
		# for l in self.additional_blocks:
		# 	x = l(x)
		# 	detection_feed.append(x)
		# locs_tmp, confs_tmp = self.bbox_view(detection_feed, self.loc, self.conf)
		# locs_l.append(locs_tmp)
		# confs_l.append(confs_tmp)

		# plan 2: independent input
		for i in range(self.split_factor):
			x = xs[i]
			detection_feed = [x]
			for l in self.additional_blocks:
				x = l(x)
				detection_feed.append(x)
			locs_tmp, confs_tmp = self.bbox_view(detection_feed, self.loc, self.conf)
			locs_l.append(locs_tmp)
			confs_l.append(confs_tmp)

		return locs_l, confs_l


if __name__ == "__main__":
	from .utils import get_the_number_of_params
	from .thop import profile, clever_format
	# model = ResNetv3(split_factor=2,
	# 					pretrained_dir='/home/zhaoshuai/models/pretrained/wide_resnet50_2_split2_imagenet_256_03/model_best.pth.tar')
	# 					# pretrained_dir='/home/zhaoshuai/models/pretrained/wide_resnet50_2_split1_imagenet_256_01/model_best.pth.tar')
	# print('The number of parameters are {}'.format(get_the_number_of_params(model)))
	# x = torch.rand(1, 3, 300, 300)
	# y = model(x)
	# for y_ in y:
	# 	print(y_.shape)

	# ssd = SSDv3(split_factor=1,
	# 				# pretrained_dir='/home/zhaoshuai/models/pretrained/wide_resnet50_2_split2_imagenet_256_03/model_best.pth.tar',
	# 				arch='wide_resnet50_3',
	# 				res_blocks=2)
	# ssd = SSDv4(split_factor=1,
	# 				pretrained_dir='/home/zhaoshuai/models/pretrained/wide_resnet50_2_split1_imagenet_256_01/model_best.pth.tar',
	# 				res_blocks=2)
	ssd = SSDv2(split_factor=2,
					# pretrained_dir='/home/zhaoshuai/models/pretrained/wide_resnet50_2_split2_imagenet_256_03/model_best.pth.tar',
					arch='resnext101_64x4d',
					res_blocks=2)

	x = torch.rand(2, 3, 300, 300)
	macs, params = profile(ssd, inputs=(x, ))
	macs = macs / 2
	macs, params = clever_format([macs, params], "%.3f")
	print(macs, params)

	# print('The number of parameters is {}'.format(get_the_number_of_params(ssd)))
	# print('The number of feature extractor parameters is {}'.format(get_the_number_of_params(ssd.feature_extractor)))
	# x = torch.rand(2, 3, 300, 300)
	# y1, y2 = ssd(x)

	# print(y1[0].shape)
	# # print(y1[1].shape)
	# print(y2[0].shape)
	# print(y2[1].shape)
