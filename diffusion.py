import os
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch
from PIL  import Image 

import torchvision.transforms as transforms
import numpy as np 


import sys
from io import StringIO
from pytorch_lightning import seed_everything
import utils_img

import warnings

from diffusers import DPMSolverMultistepScheduler

from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path

import utils
import utils_img
import utils_model

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider


# 抑制所有警告信息的输出
warnings.filterwarnings('ignore')
 
# 这里是可能会产生警告的代码
# warnings.warn("This is a warning message!")

def  img2img(imgs):
    c, h,w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
    # #抑制输出， 重定向 stdout 到 devnull
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')
    warnings.filterwarnings('ignore')
    
    # 以下代码引入SDXL模型
    model= "/ssd-sata1/hqq/stable_signature/stable-diffusion-2-1-base"
    pipeline_img2img = AutoPipelineForImage2Image.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    seed_everything(seed=20)
    pipeline_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipeline_img2img.scheduler.config)

    #pipeline_img2img.eval()
    #pipeline_text2image.model.config.image_size = imgs.shape[-1]
    prompt = "A photo"
    generator = torch.Generator("cuda").manual_seed(0)


    for i in range(imgs.shape[0]):
        img=imgs[i]
        img2 = pipeline_img2img(prompt, image=img, strength=0.8, guidance_scale=10.5,height=imgs.shape[-2],
                                width=imgs.shape[-1],generator=generator, num_inference_steps=20).images[0]

        to_tensor=transforms.ToTensor()
        img2_tensor=utils_img.normalize_rgb(to_tensor(img2))  ## c h w , 且被归一到了-1到 1区间内   ###这一行之后没有了梯度。
        img2_tensor=img2_tensor.unsqueeze(0)
        if i==0:
            output= img2_tensor
        else:
            output= torch.cat((output,img2_tensor),dim=0)
    
    #output=output.requires_grad_(True)

    return  output 


def img2img(imgs):

    #print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    ldm_config='LDM_configs/v1-inference.yaml'
    ldm_ckpt=' v2-1_768-nonema-pruned.ckpt'
    #config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(ldm_config, ldm_ckpt)
    ldm_ae.eval()
    ldm_ae.to('cuda')

    
    # encode images    
    imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
    imgs_z = imgs_z.mode()

    

    # decode latents with original and finetuned decoder
    imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w    ###decode出来的图片
    #imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w    用finetuned decoder   decode出来的图片
