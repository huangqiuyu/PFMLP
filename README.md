## PFMLP: A Pyramid Fusion MLP for Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)



<p align="middle">
  <img src="figures/pfmlp.pdf" height="300" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/flops_acc.pdf" height="300" />
</p>


## Updates

- (06/06/2024) Initial release.



## Model Zoo

We provide PFMLP models pretrained on ImageNet 2012.

| Model                | Parameters | FLOPs    | Top 1 Acc. | Download |
| :------------------- | :--------- | :------- | :--------- | :------- |
| PFMLP-N           | 19M        |  2.4G    |  82.0%     |[model](https://github.com/huangqiuyu/PFMLP/releases/download/v0.1/PFMLP_Nano.pth)|
| PFMLP-T           | 33M        |  4.2G    |  83.1%     |[model](https://github.com/huangqiuyu/PFMLP/releases/download/v0.1/PFMLP_Tiny.pth)|
| PFMLP-S           | 50M        |  6.6G    |  83.8%     |[model](https://github.com/huangqiuyu/PFMLP/releases/download/v0.1/PFMLP_Small.pth)|
| PFMLP-B           | 71M        |  9.4G    |  84.1%     |[model](https://github.com/huangqiuyu/PFMLP/releases/download/v0.1/PFMLP_Base.pth)|

## Usage


### Install

- PyTorch 1.7.0+ and torchvision 0.8.1+
- timm
- lmdb
- thop (optional, for FLOPs calculation)
```
pip install timm lmdb thop
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation
To evaluate a pre-trained PFMLP_Tiny on ImageNet val with a single GPU run:
```
python main.py --eval true --model tiny --resume path/to/PFMLP_Tiny.pth --data-path /path/to/imagenet
```


### Training

To train PFMLP_Tiny on ImageNet on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model tiny --epochs 300 --batch-size 128 --update_freq 4 --use_amp true --data-path /path/to/imagenet --output_dir /path/to/save
```


## License

PFMLP is released under MIT License.
