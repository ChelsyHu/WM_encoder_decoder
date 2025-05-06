import os
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch
from PIL  import Image 

import torchvision.transforms as transforms
import numpy as np 

def  img2img(imgs):
    c, h,w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]

    # 以下代码引入SDXL模型
    model= "/ssd-sata1/hqq/stable_signature/stable-diffusion-2-1-base"
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    #pipeline_text2image.model.config.image_size = imgs.shape[-1]
    prompt = ""


    for i in range(imgs.shape[0]):
        img=imgs[i]*255

        img =  transforms.ToPILImage(img)
        img2 = pipeline_text2image(prompt, image=img, strength=0.8, guidance_scale=10.5,height=imgs.shape[-2],width=imgs.shape[-1]).images[0]

        totensor=transforms.PILToTensor()
        img2_tensor= totensor(img2)/255  ## c h w , 且被归一到了0-1区间内
        img2_tensor=img2_tensor.unsqueeze(0)
        if i==0:
            output= img2_tensor
        else:
            output= torch.cat((output,img2_tensor),dim=0)


    print(output)
    print(type(output))
    output=torch.tensor(output)
    return  output 


imgs=torch.rand(2, 3, 128,128)

output=img2img(imgs)
print(output.shape)


