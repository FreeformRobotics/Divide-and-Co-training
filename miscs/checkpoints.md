# Checkpoints

Each zip file contains 4 types of files

* a checkpoint of the model, typically, named as `model_best.pth.tar`
* the md5 of the checkpoint
* a hyper-parameter json file, typically, named as `hparams_train.json`
* `tensorboard` log file, you can use `tensorboard` to visualize the log. It is in the `val` directory within the zip file.


## ImageNet

| Methods                   		| Top-1./Top-5 Acc (%) 	| # MParams/GFLOPs    | Checkpoints  |
|-----------------------------------|-----------------------|---------------------|--------------|
| ResNet-50, 224px           		| 78.84 / 94.47         | 25.7 / 5.5       	  | [resnet50_split1_imagenet_256_06](TODO) |
| ResNet-110, 224px      		 	| 80.16 / 94.54         | 44.8 / 9.2          | [resnet101_split1_imagenet_256_01](TODO) |
| WRN-50-2, 224px           		| 80.66 / 95.16         | 68.9 / 12.8         | [wide_resnet50_2_split1_imagenet_256_01](TODO) |
| WRN-50-2, S=2, 224px      		| 79.64 / 94.82         | 51.4 / 10.9         | [wide_resnet50_2_split2_imagenet_256_02](TODO) | 
| WRN-50-3, 224px           		| 80.74 / 95.40         | 135.0 / 23.8        | [wide_resnet50_3_split1_imagenet_256_01](TODO) |
| WRN-50-3, S=2, 224px      		| 81.42 / 95.62         | 138.0 / 25.6        | [wide_resnet50_3_split2_imagenet_256_02](TODO) | 
| ResNeXt-101, 64x4d, 224px 		| 81.57 / 95.73         | 83.6 / 16.9         | [resnext101_64x4d_split1_imagenet_256_01](TODO) |
| ResNeXt-101, 64x4d, S=2, 224px  	| 82.13 / 95.98         | 88.6 / 18.8         | [resnext101_64x4d_split2_imagenet_256_02](TODO) |
| EfficientNet-B7, 320px 			| 81.83 / 95.78         | 66.7 / 10.6         | [efficientnetb7_split1_imagenet_128_03](TODO) |
| EfficientNet-B7, S=2, 320px  		| 82.74 / 96.30         | 68.2 / 10.5         | [efficientnetb7_split2_imagenet_128_02](TODO) | 
| SE-ResNeXt-101, 64x4d, S=2, 416px | 83.34 / 96.61         | 98.0 / 61.1         | [se_resnext101_64x4d_split2_imagenet_128_02](TODO) |


## CIFAR-100

| Methods                   | Top-1. Acc (%) | # MParams/GFLOPs    | Checkpoints  |
|---------------------------|----------------|---------------------|--------------|  
| WRN-40-10, S=1            | 83.98          | 55.9 / 8.08         | [TODO](TODO) |
| WRN-40-10, S=2            | 85.91          | 54.8 / 7.94         | [TODO](TODO) |
| WRN-40-10, S=4            | 86.90          | 56.0 / 8.12         | [TODO](TODO) |
| DenseNet-BC-190, S=1      | 85.90          | 25.8 / 9.39         | [TODO](TODO) |
| DenseNet-BC-190, S=2      | 87.36          | 25.5 / 9.24         | [TODO](TODO) |
| DenseNet-BC-190, S=4      | 87.44          | 26.3 / 9.48         | [TODO](TODO) |
| PyramidNet-272, S=1       | 88.98          | 26.8 / 4.55         | [TODO](TODO) |
| PyramidNet-272, S=2       | 89.25          | 28.9 / 5.24         | [TODO](TODO) |
| PyramidNet-272, S=4       | 89.46          | 32.8 / 6.33         | [TODO](TODO) |


## CIFAR-10

| Methods                   | Top-1. Acc (%) | # MParams/GFLOPs    | Checkpoints  |
|---------------------------|----------------|---------------------|--------------|
| WRN-28-10 				| 97.59          | 36.5 / 5.25         | [TODO](TODO) |
| WRN-28-10, S=2            | 98.19          | 35.8 / 5.16         | [TODO](TODO) |
| WRN-28-10, S=4            | 98.32          | 36.5 / 5.28         | [TODO](TODO) |
| WRN-40-10 				| 97.81          | 55.8 / 8.08         | [TODO](TODO) |
| WRN-40-10, S=4            | 98.38          | 55.9 / 8.12         | [TODO](TODO) |
| Shake-Shake 26 2x96d	 	| 98.00          | 26.2 / 3.78         | [TODO](TODO) |
| Shake-Shake 26 2x96d, S=2 | 98.25          | 23.3 / 3.38         | [shake_resnet26_2x96d_split2_cifar10_128_12](TODO) |
| Shake-Shake 26 2x96d, S=4 | 98.31          | 26.3 / 3.81         | [shake_resnet26_2x96d_split4_cifar10_128_09](TODO) |
| PyramidNet-272            | 98.67          | 26.2 / 4.55         | [TODO](TODO) |
| PyramidNet-272, S=4       | 98.71          | 32.6 / 6.33         | [TODO](TODO) |