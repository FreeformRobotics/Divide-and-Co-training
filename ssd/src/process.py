"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import time
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from pycocotools.cocoeval import COCOeval
from .distributed import is_master


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, device, scaler=None, args=None):
	model.train()
	num_iter_per_epoch = len(train_loader)
	progress_bar = tqdm(train_loader)

	for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
		if torch.cuda.is_available():
			img = img.to(device)
			gloc = gloc.to(device)
			glabel = glabel.to(device)
		gloc = gloc.transpose(1, 2).contiguous()

		optimizer.zero_grad()
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			ploc_l, plabel_l = model(img)
			loss_l = []
			for s in range(len(ploc_l)):
				ploc, plabel = ploc_l[s].float(), plabel_l[s].float()
				loss_l.append(criterion(ploc, plabel, gloc, glabel))
			loss = torch.sum(torch.stack(loss_l))	
	
		if scaler is not None:
			scaler.scale(loss).backward()
			if args.clip_grad_norm is not None:
				# we should unscale the gradients of optimizer's assigned params if do gradient clipping
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			scaler.step(optimizer)
			scaler.update()
		else:
			# backward prop.
			loss.backward()
			if args.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			optimizer.step()		

		if is_master():
			progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))		
			if writer is not None:
				writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

	scheduler.step()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold, device, args=None):
	model.eval()
	detections = []
	category_ids = test_loader.dataset.coco.getCatIds()
	for nbatch, data in enumerate(test_loader):
		img, img_id, img_size = data[0], data[1], data[2]
		print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
		if torch.cuda.is_available(): img = img.to(device)
		with torch.no_grad():
			# Get predictions, list of [N, 4, 8732], [N, n_classes, 8732]
			ploc_l, plabel_l = model(img)

			for idx in range(img.shape[0]):
				# mimic batch size 1
				ploc_i =  [ploc[idx, :, :].unsqueeze(0) for ploc in ploc_l]
				plabel_i = [plabel[idx, :, :].unsqueeze(0) for plabel in plabel_l]
				try:
					result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200 * len(ploc_l))[0]
				except:
					print("No object detected in idx: {}".format(idx))
					continue

				height, width = img_size[idx][0].item(), img_size[idx][1].item(),
				loc, label, prob = [r.cpu().numpy() for r in result]
				for loc_, label_, prob_ in zip(loc, label, prob):
					detections.append([img_id[idx].item(), loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
									(loc_[3] - loc_[1]) * height, prob_,
									category_ids[label_ - 1]])

	detections = np.array(detections, dtype=np.float32)
	coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	if writer is not None:
		writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)

	with open(os.path.join(args.save_folder, 'log.txt'), 'a+') as f:
		f.write(str(coco_eval.stats))
		f.write('\nMean Average Precision (mAP): {:.3f} at {} epoch\n'.format(coco_eval.stats[0], epoch))
