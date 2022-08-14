# Dividing For Object Detection

##  Table of Contents

<!--ts-->
* [Benchmarks and Checkpoints](../miscs/checkpoints.md)
* [Installation](#Installation)
* [Training](#Training)
* [Evaluation](#Evaluation)
<!--te-->

## Installation

* **Install dependencies via docker**

Please install PyTorch-1.9.0 and Python3.6+. PyTorch-1.6.0+ should work.
Only PyTorch-1.6.0+ supports built-in AMP training.

We recommend you to use our established PyTorch docker image:
[zhaosssss/torch_lab:1.9.6](https://hub.docker.com/r/zhaosssss/torch_lab).
```
docker pull zhaosssss/torch_lab:1.9.6
```
If you have not installed docker, see https://docs.docker.com/. 


### Prepare data

Generally, directories are organized as following:
```
${HOME}
├── dataset             (save the dataset) 
│   │
│   ├── coco            (dir of cooc dataset)
│
├── models              (save the output checkpoints)
│
├── github              (save the code)
│   │   
│   └── splitnet        (the splitnet code repository)
│       │
│       └── ssd
│           │   
│           ├── src
│           ├── train.sh
│           └── train.py 
...
```

- Download [The COCO 2017 dataset](https://cocodataset.org/#download),
put them in the `dataset/coco` directory.
```
coco
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
│── train2017
└── val2017 
```

- `cd` to `github` directory and clone the `Divide-and-Co-training` repo.
For brevity, rename it as `splitnet`.


## Training

See `train.sh` for detailed information.
Before start training, you should specify some variables in the `train.sh`.

For example:

- `arch`, the architecture you want to use.


You can find more information about the arguments of the code in `train.py`.
```
python train.py --help
usage: train.py [-h] [--data_path DATA_PATH] [--save_folder SAVE_FOLDER]
                [--model {ssd,ssdv2}]
                [--arch {wide_resnet50_2,wide_resnet50_3,resnext101_64x4d}]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--multistep [MULTISTEP [MULTISTEP ...]]]
                [--precision {amp,fp16,fp32}]
                [--clip_grad_norm CLIP_GRAD_NORM] [--lr LR]
                [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                [--nms_threshold NMS_THRESHOLD] [--num_workers NUM_WORKERS]
                [--dist_backend DIST_BACKEND] [--world_size WORLD_SIZE]
                [--local_rank LOCAL_RANK] [--init_method INIT_METHOD]
                [--split_factor SPLIT_FACTOR]
                [--pretrained_dir PRETRAINED_DIR] [--resume RESUME]
                [--eval_freq EVAL_FREQ] [--eval EVAL]
                [--res_blocks RES_BLOCKS]

Implementation of SSD

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        the root folder of dataset
  --save_folder SAVE_FOLDER
                        path to folder containing model checkpoint file
  --model {ssd,ssdv2}   ssdv2 for normal ssd, ssdv2 for SSD with dividing and
                        co-training
  --arch {wide_resnet50_2,wide_resnet50_3,resnext101_64x4d}
                        ssd-resnet50 or ssdlite-mobilenetv2
  --epochs EPOCHS       number of total epochs to run
  --batch_size BATCH_SIZE
                        number of samples for each iteration
  --multistep [MULTISTEP [MULTISTEP ...]]
                        epochs at which to decay learning rate
  --precision {amp,fp16,fp32}
                        Floating point precition.
  ...
```


After you set all the arguments properly, you can simply run
```
bash train.sh
```
to start training.


### GPU memory usage
At least 1 3090 NVIDIA GPU is needed.

## Evaluation

Set `eval=1` in the `train.sh` Then run 
```
bash train.sh
```
You will get things like
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.330
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
```



