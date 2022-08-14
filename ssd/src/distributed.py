# coding=utf-8
# some utils for distributed training
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py
import os
import torch
import random
import numpy as np
import torch.distributed as dist


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_master():
	"""check if current process is the master"""
	return get_rank() == 0


def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	import builtins as __builtin__

	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop("force", False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print


def set_random_seed(seed=None, FLAGS=None):
	"""set random seed"""
	if seed is None:
		seed = getattr(FLAGS, 'random_seed', 3407)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def init_distributed_mode(args, ngpus_per_node, gpu):
	"""initialize for distributed training"""

	if args.distributed:
		print("INFO: [CUDA] Initialize process group for distributed training")
		global_rank = args.local_rank * ngpus_per_node + gpu
		print("INFO: [CUDA] Use [GPU: {} / Global Rank: {}] for training, "
						"init_method {}, world size {}".format(gpu, global_rank, args.init_method, args.world_size))
		# set device before init process group
		# Ref: https://github.com/pytorch/pytorch/issues/18689
		torch.cuda.set_device(args.gpu)	
		torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
												world_size=args.world_size, rank=global_rank)
		torch.distributed.barrier(device_ids=[args.gpu])
		setup_for_distributed(global_rank == 0)

	else:
		args.local_rank = gpu
		global_rank = 0
		print("Use [GPU: {}] for training".format(gpu))

	return global_rank


def init_distributed_mode_auto(args):
	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		args.rank = int(os.environ["RANK"])
		args.world_size = int(os.environ["WORLD_SIZE"])
		args.gpu = int(os.environ["LOCAL_RANK"])
	elif "SLURM_PROCID" in os.environ:
		args.rank = int(os.environ["SLURM_PROCID"])
		args.gpu = args.rank % torch.cuda.device_count()
	elif hasattr(args, "rank"):
		pass
	else:
		print("Not using distributed mode")
		args.distributed = False
		if torch.cuda.is_available():
			args.gpu = 0
		else:
			args.gpu = None
		return

	args.distributed = True

	torch.cuda.set_device(args.gpu)
	args.dist_backend = "nccl"
	print(f"| distributed init (rank {args.rank}): {args.init_method}", flush=True)
	torch.distributed.init_process_group(
		backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank
	)
	torch.distributed.barrier(device_ids=[args.gpu])
	setup_for_distributed(args.rank == 0)
