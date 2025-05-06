# 以下代码为程序运行进行设置

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from diffusers import AutoPipelineForText2Image

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

import torch

# 以下代码引入SDXL模型
model= "/ssd-sata1/hqq/stable_signature/stable-diffusion-2-1-base"

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#          model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

# 这里使用from_pipe方法加载checkpoint,避免额外消耗内存
pipeline = AutoPipelineForImage2Image.from_pretrained(
         model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# 以下代码加载原始图像

url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"

init_image = load_image(url)
init_image.save('img_outputs/init.png')

# 以下代码通过原始图像和提示词，通过图生图的方式生成新图像

prompt = " "
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images

print(type(image))
image=image[0]
image.save('img_outputs/img2img.png')
