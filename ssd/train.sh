#!/bin/bash
export PYTHONUNBUFFERED="True"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using local machine for training"
s_gpus=${gpus//,/}
num_gpus=${#s_gpus}

data_path=${HOME}/dataset1/coco
epochs=65
lr=2.6e-3
weight_decay=5e-4
batch_size=32
precision=fp32
num_workers=12
model=ssd
arch=wide_resnet50_2
pretrained_dir=None
split_factor=1
eval_freq=5

runfile=./train.py


for num in 02 06
do
case ${num} in
   02 )
		# 2 X 3090 GPUs
		batch_size=64
		precision=amp
		arch=wide_resnet50_2
		model=ssdv2
		split_factor=1
		pretrained_dir='${HOME}/models/pretrained/wide_resnet50_2_split1_imagenet_256_01/model_best.pth.tar'
    	eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_02/SSD.pth
		;;
   06 )
		# 2 X 3090 GPUs
		batch_size=32
		precision=amp
		arch=wide_resnet50_2
		model=ssdv2
		split_factor=2
		# pretrained_dir="${HOME}/models/pretrained/wide_resnet50_2_split2_imagenet_256_03/model_best.pth.tar"
    	eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_06/SSD.pth
		;;
   07 )
		# 2 X 3090 GPUs
		batch_size=32
		precision=amp
		arch=wide_resnet50_3
		model=ssdv2
		split_factor=1
		pretrained_dir="${HOME}/models/pretrained/wide_resnet50_3_split1_imagenet_256_01/model_best.pth.tar"
		eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_07/SSD.pth
		;;
   09 )
		# 2 X 3090 GPUs
		batch_size=32
		precision=amp
		arch=wide_resnet50_3
		model=ssdv2
		split_factor=2
		pretrained_dir="${HOME}/models/pretrained/wide_resnet50_3_split2_imagenet_256_02/model_best.pth.tar"
    	eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_09/SSD.pth
		;;
   14 )
		# 2 X 3090 GPUs
		batch_size=32
		precision=amp
		arch=resnext101_64x4d
		model=ssdv2
		split_factor=1
		pretrained_dir="${HOME}/models/pretrained/resnext101_64x4d_split1_imagenet_256_01/model_best.pth.tar"
		eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_14/SSD.pth
		;;
   15 )
		# 2 X 3090 GPUs
		batch_size=24
		precision=amp
		arch=resnext101_64x4d
		model=ssdv2
		split_factor=2
		pretrained_dir="${HOME}/models/pretrained/resnext101_64x4d_split2_imagenet_256_02/model_best.pth.tar"
		eval=1
		resume=${HOME}/models/ssd/ssd300_coco_ssdv2_15/SSD.pth
		;;
	* )
		;;
esac


output_dir=${HOME}/models/ssd/ssd300_coco_${model}_${num}
# python ${runfile} --data_path ${data_path} \
python -m torch.distributed.launch --nproc_per_node=${num_gpus} --max_restarts=0 ${runfile} --data_path ${data_path} \
					--save_folder ${output_dir} \
					--num_workers ${num_workers} \
					--batch_size ${batch_size} \
					--precision ${precision} \
					--lr ${lr} \
					--epochs ${epochs} \
					--model ${model} \
					--arch ${arch} \
					--pretrained_dir ${pretrained_dir} \
					--split_factor ${split_factor} \
					--eval_freq ${eval_freq} \
					--eval ${eval} \
					--resume ${resume} \
					--weight_decay ${weight_decay}

done
