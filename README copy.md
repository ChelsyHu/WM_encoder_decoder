# Image Watermarking with Revisited HiDDeN

[`Colab demo`](https://colab.research.google.com/github/facebookresearch/stable_signature/blob/master/hidden/notebooks/demo.ipynb)
(for using the pre-trained networks for traditional image watermarking.)

This repository is heavily based on the paper [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937).
The main differences are:
- there is no adversarial network, nor any image loss,
- robustness with regards to JPEG is done by stopping the gradient (see [Towards Robust Deep Hiding Under Non-Differentiable Distortions for Practical Blind Watermarking](https://dl.acm.org/doi/10.1145/3474085.3475628)).

Another implementation is available at [ando-khachatryan/HiDDeN](https://github.com/ando-khachatryan/HiDDeN).


## Setup

### Requirements

This codebase has been developed with python version 3.8, PyTorch version 1.12.0, CUDA 11.3.
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit=11.3
```

To install the remaining dependencies with pip, run:
```cmd
pip install -r requirements.txt
```

### Data

The paper uses the [COCO](https://cocodataset.org/) dataset.


## Usage


### 

### Training

The main script is in `main.py`. It can be used to train the encoder and decoder networks.

To run it on one GPU, use the following command:
```bash
torchrun --nproc_per_node=1 main.py --dist False
```

To run it on multiple GPUs, use the following command:
```bash
torchrun --nproc_per_node=$GPUS$ main.py --local_rank 0
```

### Options


<details>
<summary><span style="font-weight: normal;">Experiment Parameters</span></summary>

- `--train_dir`: Path to the directory containing the training data. Default: "path/to/train"
- `--val_dir`: Path to the directory containing the validation data. Default: "path/to/val"
- `--output_dir`: Output directory for logs and images. Default: "output/"
- `--verbose`: Verbosity level for output during training. Default: 1
- `--seed`: Random seed. Default: 0
</details>

<details>
<summary><span style="font-weight: normal;">Marking Parameters</span></summary>

- `--num_bits`: Number of bits in the watermark. Default: 32
- `--redundancy`: Redundancy of the watermark in the decoder (the output is bit is the sum of redundancy bits). Default: 1
- `--img_size`: Image size during training. Having a fixed image size during training improves efficiency thanks to batching. The network can generalize (to a certain extent) to arbitrary resolution at test time. Default: 128
</details>

<details>
<summary><span style="font-weight: normal;">Encoder Parameters</span></summary>

- `--encoder`: Encoder type (e.g., "hidden", "dvmark", "vit"). Default: "hidden"
- `--encoder_depth`: Number of blocks in the encoder. Default: 4
- `--encoder_channels`: Number of channels in the encoder. Default: 64
- `--use_tanh`: Use tanh scaling. Default: True
</details>

<details>
<summary><span style="font-weight: normal;">Decoder Parameters</span></summary>

- `--decoder`: Decoder type (e.g., "hidden"). Default: "hidden"
- `--decoder_depth`: Number of blocks in the decoder. Default: 8
- `--decoder_channels`: Number of channels in the decoder. Default: 64
</details>

<details>
<summary><span style="font-weight: normal;">Training Parameters</span></summary>

- `--bn_momentum`: Momentum of the batch normalization layer. Default: 0.01
- `--eval_freq`: Frequency of evaluation during training (in epochs). Default: 1
- `--saveckp_freq`: Frequency of saving checkpoints (in epochs). Default: 100
- `--saveimg_freq`: Frequency of saving images (in epochs). Default: 10
- `--resume_from`: Checkpoint path to resume training from.
- `--scaling_w`: Scaling of the watermark signal. Default: 1.0
- `--scaling_i`: Scaling of the original image. Default: 1.0
</details>

<details>
<summary><span style="font-weight: normal;">Optimization Parameters</span></summary>

- `--epochs`: Number of epochs for optimization. Default: 400
- `--optimizer`: Optimizer to use (e.g., "Adam"). Default: "Adam"
- `--scheduler`: Learning rate scheduler to use (ex: "CosineLRScheduler,lr_min=1e-6,t_initial=400,warmup_lr_init=1e-6,warmup_t=5"). Default: None
- `--lambda_w`: Weight of the watermark loss. Default: 1.0
- `--lambda_i`: Weight of the image loss. Default: 0.0
- `--loss_margin`: Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. Default: 1.0
- `--loss_i_type`: Loss type for image loss ("mse" or "l1"). Default: 'mse'
- `--loss_w_type`: Loss type for watermark loss ("bce" or "cossim"). Default: 'bce'
</details>

<details>
<summary><span style="font-weight: normal;">Loader Parameters</span></summary>

- `--batch_size`: Batch size for training. Default: 16
- `--batch_size_eval`: Batch size for evaluation. Default: 64
- `--workers`: Number of workers for data loading. Default: 8
</details>

<details>
<summary><span style="font-weight: normal;">Attenuation Parameters</span></summary>

Additonally, the codebase allows to train with a just noticeable difference map (JND) to attenuate the watermark signal in the perceptually sensitive regions of the image.
This can also be added at test time only, at the cost of some accuracy.
- `--attenuation`: Attenuation type. Default: None
- `--scale_channels`: Use channel scaling. Default: True
</details>

<details>
<summary><span style="font-weight: normal;">Data Augmentation Parameters</span></summary>

- `--data_augmentation`: Type of data augmentation to use at marking time ("combined", "kornia", "none"). Default: "combined"
- `--p_crop`: Probability of the crop augmentation. Default: 0.5
- `--p_res`: Probability of the resize augmentation. Default: 0.5
- `--p_blur`: Probability of the blur augmentation. Default: 0.5
- `--p_jpeg`: Probability of the JPEG compression augmentation. Default: 0.5
- `--p_rot`: Probability of the rotation augmentation. Default: 0.5
- `--p_color_jitter`: Probability of the color jitter augmentation. Default: 0.5
</details>

<details>
<summary><span style="font-weight: normal;">Distributed Training Parameters</span></summary>

- `--debug_slurm`: Enable debugging for SLURM.
- `--local_rank`: Local rank for distributed training. Default: -1
- `--master_port`: Port for the master process. Default: -1
- `--dist`: Enable distributed training. Default: True
</details>



### Example

For instance the following command line reproduces the hidden extractor with same parameters as in the paper:
```cmd
torchrun --nproc_per_node=8 main.py \
  --val_dir path/to/coco/test2014/ --train_dir path/to/coco/train2014/ --output_dir output --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 
```


CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 main.py \
  --val_dir datasets/coco_dataset/test2014  --train_dir datasets/coco_dataset/train2014 --output_dir output/output_bit/7  --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --local_rank 0  --dist False  --workers 2


CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 main4.py \
  --val_dir datasets/val_dir  --train_dir datasets/train_dir --output_dir output/output_bit/9_mytest  --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-5,t_initial=300,warmup_lr_init=1e-4,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 0.0 --p_res 0.0 --p_jpeg 0.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --local_rank 0  --dist True  --workers 2


This should create a folder `output` with the checkpoints, logs and images.
The logs during training are:
[`logs.txt`](https://dl.fbaipublicfiles.com/ssl_watermarking/logs_replicate.txt) - [`console.stdout`](https://dl.fbaipublicfiles.com/ssl_watermarking/logs_replicate.stdout).

The resulting checkpoints have approximately the same performance as in the paper:
[`hidden_replicate.pth`](https://dl.fbaipublicfiles.com/ssl_watermarking/hidden_replicate.pth) - [`hidden_replicate_whit.torchscript.pth`](https://dl.fbaipublicfiles.com/ssl_watermarking/hidden_replicate_whit.torchscript.pt).
(Robustness to JPEG is a bit worse, because the augmentation implementation differ a bit from the paper: this could be fixed by increasing the JPEG augmentation probability).



## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={arXiv preprint arXiv:2303.15435},
  year={2023}
}
```


2024年7月13日 星期六 总结我的代码：
main.py  models.py  是原来的作者写的关于比特串的注入和提取器训练的代码。（我自己基于它训练出来的ckpt模型效果 目前没有HIdden文章中给出的模型文件好使）


CUDA_VISIBLE_DEVICES=1 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 8 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=0 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 8 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=3 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 8 --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=3 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 12 --epochs 350 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=8  main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 12 --epochs 350 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --local_rank -1 --dist True --workers 2



torchrun --nproc_per_node=8 main.py \
  --val_dir path/to/coco/test2014/ --train_dir path/to/coco/train2014/ --output_dir output --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 


10.0  1   0.04这种lambda设置好像不错。
lr设置为0.0004好像不错


CUDA_VISIBLE_DEVICES=2 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1

## 2024-10-31
### main7.py model7.py  可以达到有loss_w, loss_i, loss_latents
CUDA_VISIBLE_DEVICES=7 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1 --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1


#### 后续可以看下， 分步训练

## 还可以看一下如何修改loss3

这个应该效果还好：考虑到了（1-loss3）
CUDA_VISIBLE_DEVICES=7 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1 --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1


      CUDA_VISIBLE_DEVICES=6  python  main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts/3 --eval_freq 5   --img_size 256 --num_bits 48  --batch_size 16 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5     --optimizer Adam,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False  --local_rank 0



CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_specific_WikiArt3   --output_dir  ckpts/1    --img_size 512 --num_bits 48   --batch_size_eval 32 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0  


## 水印
100101101011101100100110010100001101011011100110

## 2024-11-23
正在跑
CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_specific_WikiArt3   --output_dir  ckpts/1    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0  
（进行imgs, imgs_w， imgs_generated, imgs_w_generated的生成）

之后需要做fid的计算（imgs_generated，imgs_w_generated之间的 ）


我想继续优化原来的checkpoint文件。




python main_new.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 8 --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False  --local_rank 0





CUDA_VISIBLE_DEVICES=1,2  python -m torch.distributed.launch --nproc-per-node=2  \
  test_bit.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new_copy_2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist True 


## 测试原来的ckpt(300回合)对新测试数据集的适应能力（50000）
CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new_copy_2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


## 

      CUDA_VISIBLE_DEVICES=6  python  main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts/new_copy --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 16 --epochs 400   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5     --optimizer Adam,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False  --local_rank 0


      CUDA_VISIBLE_DEVICES=2 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1 

  
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc-per-node=4  main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist True  --local_rank 0  --master_port -1



CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node=4 main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist True  



CUDA_VISIBLE_DEVICES=1 python main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False 

首先用 0.001 的学习率， lambda_w 10， lambda_latents_b 1
然后用 0.0005的学习率，  lambda_w 10， lambda_latents_b 2
用 0.0001， lambda_w 10, lambda_latents_b 2 (331开始的)

0.00005 336开始


## 测试
CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_small_WikiArt3   --output_dir  ckpts/2    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0

CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --test_dir datasets/Test_small_WikiArt3 \
  --output_dir ckpts/2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


CUDA_VISIBLE_DEVICES=6  python  test.py   --test_dir datasets/Test_small_WikiArt3   --output_dir  ckpts/0    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0

CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --test_dir datasets/Test_small_WikiArt3 \
  --output_dir ckpts/0  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


CUDA_VISIBLE_DEVICES=1  python  main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False


## 2024-12-4
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py \
  --val_dir datasets/coco_dataset/test2014 --train_dir datasets/coco_dataset/train2014  --output_dir output  --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist True --local_rank 0


CUDA_VISIBLE_DEVICES=0 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/10
--output_dir watermark/small_WikiArt3_val/10  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False



CUDA_VISIBLE_DEVICES=2 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/20
--output_dir watermark/small_WikiArt3_val/20  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False


CUDA_VISIBLE_DEVICES=3 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/30
--output_dir watermark/small_WikiArt3_val/30  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False


CUDA_VISIBLE_DEVICES=4 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/40
--output_dir watermark/small_WikiArt3_val/40  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False

CUDA_VISIBLE_DEVICES=5 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/50
--output_dir watermark/small_WikiArt3_val/50  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False

## 2024-12-30
CUDA_VISIBLE_DEVICES=3 python test_bit2.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/50
--output_dir watermark/small_WikiArt3_val/50_2  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 8 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False
