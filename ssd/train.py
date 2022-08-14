# coding=utf-8
import os
import torch
from argparse import ArgumentParser
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from src.loss import Loss
from src.model import SSD, ResNet, SSDv2
from src.process import train, evaluate
from src.transform import SSDTransformer
from src.dataset import CocoDataset, collate_fn
from src.utils import generate_dboxes, Encoder, coco_classes
from src.distributed import (init_distributed_mode_auto, get_rank, is_dist_avail_and_initialized,
											is_master, set_random_seed)


def get_args():
	parser = ArgumentParser(description="Implementation of SSD")
	
	parser.add_argument("--data_path", type=str, default="/coco",
							help="the root folder of dataset")
	
	parser.add_argument("--save_folder", type=str, default="trained_models",
							help="path to folder containing model checkpoint file")

	parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdv2"],
							help="ssdv2 for normal ssd, ssdv2 for SSD with dividing and co-training")
	parser.add_argument("--arch", type=str, default="wide_resnet50_2",
							choices=["wide_resnet50_2", "wide_resnet50_3", "resnext101_64x4d"],
							help="ssd-resnet50 or ssdlite-mobilenetv2")

	parser.add_argument("--epochs", type=int, default=65, help="number of total epochs to run")

	parser.add_argument("--batch_size", type=int, default=32, help="number of samples for each iteration")

	parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
							help="epochs at which to decay learning rate")

	parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32",
							help="Floating point precition.")

	parser.add_argument("--clip_grad_norm", default=None, type=float,
							help="the maximum gradient norm (default None)")

	parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
	parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
	parser.add_argument("--weight_decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
	parser.add_argument("--nms_threshold", type=float, default=0.5)
	parser.add_argument("--num_workers", type=int, default=8)

	parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

	parser.add_argument('--world_size', default=1, type=int,
							help='number of nodes for distributed training')

	parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

	parser.add_argument("--init_method", default="tcp://127.0.0.1:6131", type=str,
	   						help="url used to set up distributed training")
	# split factor
	parser.add_argument('--split_factor', default=1, type=int,
							help='split one big network into split_factor small networks')
	# setting about pretrained weights
	parser.add_argument("--pretrained_dir", default=os.path.expanduser("~/models/pretrained"), type=str,
							help="The pretrained directory of pretrained model")
	parser.add_argument("--resume", default=os.path.expanduser("~/models/pretrained"), type=str,
							help="resume from ckpt")
	parser.add_argument('--eval_freq', type=int, default=10, help='eval frequence')
	parser.add_argument('--eval', type=int, default=0, help='If true, only do evaluation.')
	parser.add_argument('--res_blocks', default=3, type=int,
							help='the number of features provided by the resnet backbone')

	args = parser.parse_args()

	return args


def main(opt):
	"""main worker"""
	# opt.gpu = gpu
	opt.lr = opt.lr * (opt.batch_size / 32)
	## ####################################
	# distributed training initilization
	## ####################################
	# global_rank = init_distributed_mode(opt, ngpus_per_node, gpu)
	init_distributed_mode_auto(opt)
	# here use local_rank
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank() % opt.ngpus_per_node)

	if is_master():
		if not os.path.isdir(opt.save_folder): os.makedirs(opt.save_folder)
	opt.log_path = os.path.join(opt.save_folder, 'tensorboard')
	if is_master():
		if not os.path.isdir(opt.log_path): os.makedirs(opt.log_path)
	opt.checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")
	set_random_seed(3407)
	torch.backends.cudnn.benchmark = True

	## ####################################
	# create model
	## ####################################	
	if opt.model == "ssd":
		dboxes = generate_dboxes(model="ssd")
		model = SSD(backbone=ResNet(), num_classes=len(coco_classes))
	if opt.model in ["ssdv2", "ssdv3", "ssdv4"]:
		# prior boxes, [8732, 4]
		dboxes = generate_dboxes(model="ssd")
		if opt.model == "ssdv2":
			model = SSDv2(split_factor=opt.split_factor, pretrained_dir=opt.pretrained_dir, arch=opt.arch, 
							num_classes=len(coco_classes))
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError

	if not torch.cuda.is_available():
		model.float()
		print("using CPU, this will be slow")
		# comment out the following line for debugging
		raise NotImplementedError("Only DistributedDataParallel is supported.")
	else:
		model.to(device)
		# Previously batch size and workers were global and not per GPU.
		# if args.distributed and args.use_bn_sync:
		# 	logging.info('[CUDA] Using SyncBatchNorm...')
		# 	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
		if opt.distributed:
			print("INFO: [model] creating DistributedDataParallel")
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu],
										find_unused_parameters=False)

	## ####################################
	# dataloader loading
	## ####################################
	train_set = CocoDataset(opt.data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
	train_sampler = None
	if is_dist_avail_and_initialized(): train_sampler = DistributedSampler(train_set, shuffle=True)

	train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=(train_sampler is None), drop_last=True,
								num_workers=opt.num_workers, collate_fn=collate_fn, pin_memory=True,
								sampler=train_sampler)
	test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
	test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True,
								num_workers=opt.num_workers, collate_fn=collate_fn)

	encoder = Encoder(dboxes)
	criterion = Loss(dboxes).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
								weight_decay=opt.weight_decay,
								nesterov=True)
	scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)
	# loss scaler for AMP
	scaler = GradScaler() if opt.precision == "amp" else None
	writer = None if not is_master() else SummaryWriter(opt.log_path)

	## ####################################
	#  optionally resume from a checkpoint
	## ####################################
	first_epoch = 0
	if os.path.isfile(opt.resume):
		if opt.gpu is None:
			checkpoint = torch.load(opt.resume, map_location='cpu')
		else:
			# Map model to be loaded to specified single gpu.
			loc = "cuda:{}".format(opt.gpu)
			checkpoint = torch.load(opt.resume, map_location=loc)
		sd = checkpoint["state_dict"]
		if not opt.distributed and next(iter(sd.items()))[0].startswith('module'):
			print('INFO: [resume] remove module. prefex')
			sd = {k[len('module.'):]: v for k, v in sd.items()}

		if opt.distributed and not next(iter(sd.items()))[0].startswith('module'):
			print('INFO: [resume] add module. prefex')
			sd = {'module.' + k: v for k, v in sd.items()}			

		model.load_state_dict(sd)

		first_epoch = checkpoint["epoch"] + 1
		if "scaler" in checkpoint and scaler is not None:
			print("[resume] => Loading state_dict of AMP loss scaler")
			scaler.load_state_dict(checkpoint['scaler'])
		scheduler.load_state_dict(checkpoint["scheduler"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		print('INFO: [model] resume from {}, epoch {}'.format(opt.resume, first_epoch))

	if opt.eval:
		evaluate(model, test_loader, 0, writer, encoder, opt.nms_threshold, device, args=opt)
		return

	# train and eval loop
	for epoch in range(first_epoch, opt.epochs):
		if train_sampler is not None: train_sampler.set_epoch(epoch)
		train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, device, scaler=scaler, args=opt)

		if is_master():
			if (epoch + 1) % opt.eval_freq == 0 or (epoch + 1) == opt.epochs:
				evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold, device, args=opt)

			ckpt_dict = {"epoch": epoch,
							"state_dict": model.state_dict(),
							"optimizer": optimizer.state_dict(),
							"scheduler": scheduler.state_dict()}
			if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
			torch.save(ckpt_dict, opt.checkpoint_path)
		torch.cuda.synchronize()


if __name__ == "__main__":
	opt = get_args()
	opt.distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1
	opt.ngpus_per_node = torch.cuda.device_count()

	print(opt)
	main(opt)
