######################## models.py ########################
import torch
import torch.nn as nn
from timm.models import vision_transformer

import attenuations

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs)   # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1) # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)) )   ###要这一层有什么做用吗 难道？？
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x

class ImgEmbed(nn.Module):
    """ Patch to Image Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, num_patches_w, num_patches_h):
        B, S, CKK = x.shape # ckk = embed_dim
        x = self.proj(x.transpose(1,2).reshape(B, CKK, num_patches_h, num_patches_w)) # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x

class VitEncoder(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_bits, last_tanh=True, **kwargs):
        super(VitEncoder, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.msg_linear = nn.Linear(self.embed_dim+num_bits, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=kwargs['patch_size'])

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):

        num_patches = int(self.patch_embed.num_patches**0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1) # b 1 k
        msgs = msgs.repeat(1, x.shape[1], 1) # b 1 k -> b l k
        for ii, blk in enumerate(self.blocks):
            x = torch.concat([x, msgs], dim=-1) # b l (cpq+k)
            x = self.msg_linear(x)
            x = blk(x)

        x = x[:, 1:, :] # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w

class DvmarkEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(DvmarkEncoder, self).__init__()

        transform_layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            transform_layers.append(layer)
        self.transform_layers = nn.Sequential(*transform_layers)

        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels+num_bits, channels*2)]
        for _ in range(num_blocks_scale1-1):
            layer = ConvBNRelu(channels*2, channels*2)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)

        # downsample x2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # conv layers for downsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(channels*2+num_bits, channels*4), ConvBNRelu(channels*4, channels*2)]
        for _ in range(num_blocks_scale2-2):
            layer = ConvBNRelu(channels*2, channels*2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final_layer = nn.Conv2d(channels*2, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        encoded_image = self.transform_layers(imgs) # b c h w

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1

        scale1 = torch.cat([msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)), encoded_image], dim=1) # b l+c h w
        scale1 = self.scale1_layers(scale1) # b c*2 h w

        scale2 = self.avg_pool(scale1) # b c*2 h/2 w/2
        scale2 = torch.cat([msgs.expand(-1,-1, imgs.size(-2)//2, imgs.size(-1)//2), scale2], dim=1) # b l+c*2 h/2 w/2
        scale2 = self.scale2_layers(scale2) # b c*2 h/2 w/2

        scale1 = scale1 + self.upsample(scale2) # b c*2 h w
        im_w = self.final_layer(scale1) # b 3 h w

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class EncoderDecoder(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        augmentation: nn.Module, 
        decoder:nn.Module,
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float,
        num_bits: int,
        redundancy: int
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        eval_mode: bool=False,
        eval_aug: nn.Module=nn.Identity(),
    ):
        """
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            imgs: b c h w
            msgs: b l
        """

        # encoder
        deltas_w = self.encoder(imgs, msgs) # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6 # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None,:,None,None]

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
            deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        # data augmentation
        if eval_mode:
            imgs_aug = eval_aug(imgs_w)
            fts = self.decoder(imgs_aug) # b c h w -> b d
        else:
            imgs_aug = self.augmentation(imgs_w)
            fts = self.decoder(imgs_aug) # b c h w -> b d
            
        fts = fts.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        fts = torch.sum(fts, dim=-1) # b k r -> b k

        return fts, (imgs_w, imgs_aug)

class EncoderWithJND(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
    ):
        """ Does the forward pass of the encoder only """

        # encoder
        deltas_w = self.encoder(imgs, msgs) # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels:
            aa = 1/4.6 # such that aas has mean 1
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
            deltas_w = deltas_w * aas[None,:,None,None]

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
            deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        return imgs_w


###################  attenuations.py ###################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import functional

class JND(nn.Module):
    """ Same as in https://github.com/facebookresearch/active_indexing """
    
    def __init__(self, preprocess = lambda x: x):
        super(JND, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_lum = [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.FloatTensor(kernel_lum).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.weight_lum = nn.Parameter(data=kernel_lum, requires_grad=False)

        self.preprocess = preprocess
    
    def jnd_la(self, x, alpha=1.0):
        """ Luminance masking: x must be in [0,255] """
        la = F.conv2d(x, self.weight_lum, padding=2)/32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127)) + 3
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117):
        """ Contrast masking: x must be in [0,255] """
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    def heatmaps(self, x, clc=0.3):
        """ x must be in [0,1] """
        x = 255 * self.preprocess(x)
        x = 0.299 * x[...,0:1,:,:] + 0.587 * x[...,1:2,:,:] + 0.114 * x[...,2:3,:,:]
        la =  self.jnd_la(x)
        cm = self.jnd_cm(x)
        return (la + cm - clc * torch.minimum(la, cm))/255 # b 1 h w
    



######################## demo.ipynb ############################

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

import diffusers
from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

import os 
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ############ device 

# you should run this notebook in the root directory of the hidden project for the following imports to work



def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]



class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

decoder = HiddenDecoder(
    num_blocks=params.decoder_depth, 
    num_bits=params.num_bits, 
    channels=params.decoder_channels
)
encoder = HiddenEncoder(
    num_blocks=params.encoder_depth, 
    num_bits=params.num_bits, 
    channels=params.encoder_channels
)

####
attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
encoder_with_jnd = EncoderWithJND(
    encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
)




ckpt_path = "ckpts/hidden_replicate.pth"

state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder_with_jnd = encoder_with_jnd.to(device).eval()    #### encoder_with_jnd
decoder = decoder.to(device).eval()



###################### Test ########################


# load image
img = Image.open("img_outputs/init.png").convert('RGB')
img = img.resize((512, 512), Image.BICUBIC)
img_pt = default_transform(img).unsqueeze(0).to(device)   ## 1*3*512*512

# create message
random_msg = False
if random_msg:
    msg_ori = torch.randint(0, 2, (1, params.num_bits), device=device).bool() # b k
else:
    msg_ori = torch.Tensor(str2msg("011010110101001101010111010011010100010010101101")).unsqueeze(0)
    # msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0)
msg = 2 * msg_ori.type(torch.float) - 1 # b k  ###将 0，1, 分别换成了 -1， 1
msg=msg.to(device)  ##这个是我加的  attention
msg_ori=msg_ori.to(device)


# encode
img_w = encoder_with_jnd(img_pt, msg)    ### encoder_with_jnd
clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
clip_img = torch.round(255 * clip_img)/255 
clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
#img_w=img_w.to(device)  ##

### diffusion
# loading the pipeline, and replacing the decode function of the pipe
model= "/ssd-sata1/hqq/stable_signature/stable-diffusion-2-1-base"

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
         model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")


# 这里使用from_pipe方法加载checkpoint,避免额外消耗内存
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

# 以下代码通过原始图像和提示词，通过图生图的方式生成新图像

prompt = "convert the photo to the style of Picasso"
gen_o_img = pipeline(prompt, image=img, strength=0.8, guidance_scale=10.5).images[0]
gen_w_img = pipeline(prompt, image=clip_img, strength=0.8, guidance_scale=10.5).images[0]




# psnr
psnr1 = peak_signal_noise_ratio(np.array(img), np.array(clip_img))
print(f"PSNR between original image and watermarked image: {psnr1}")


psnr2 = peak_signal_noise_ratio(np.array(gen_o_img), np.array(gen_w_img))
print(f"PSNR between images generated from original image and watermarked image: {psnr2}")



i=6             ###子目录名称修改
path=os.path.join( "img_outputs" , str(i))
os.makedirs(path, exist_ok=True)

# plot
img_pt=img_pt.cpu()

plt.figure(figsize=(4, 4))
plt.grid('off')
plt.xticks([])
plt.yticks([])
plt.title("Original Image")
plt.imshow(img)
path1=os.path.join(path,'ori_image.png')
plt.savefig(path1)

plt.figure(figsize=(4, 4))
plt.grid('off')
plt.xticks([])
plt.yticks([])
plt.title("Watermarked Image")
plt.imshow(clip_img) ##clip_image
path1=os.path.join(path,'w_image.png')
plt.savefig(path1) 


diff = np.abs(np.asarray(img).astype(int) - np.asarray(clip_img).astype(int)) / 255 * 10
plt.figure(figsize=(4, 4))
plt.grid('off')
plt.xticks([])
plt.yticks([])
plt.title("Difference")
plt.imshow(diff)
path1=os.path.join(path,'diff.png')
plt.savefig(path1) 


plt.figure(figsize=(4, 4))
plt.grid('off')
plt.xticks([])
plt.yticks([])
plt.title("Image genetated from original image")
plt.imshow(gen_o_img)
path1=os.path.join(path,'gen_o_img.png')
plt.savefig(path1) 

plt.figure(figsize=(4, 4))
plt.grid('off')
plt.xticks([])
plt.yticks([])
plt.title("Image generated from watermarked Image")
plt.imshow(gen_w_img)
path1=os.path.join(path,'gen_w_img.png')
plt.savefig(path1) 

print('\n\n')



# decode
print('Decode the watermarked image:')
ft = decoder(default_transform(clip_img).unsqueeze(0).to(device))   ###decoder的预测结果 为一个个 浮点数
decoded_msg = ft > 0 # b k -> b k      ##根据正负值 分为0-1字符串
accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k
print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")

print('\n')
print('Decode the generated images from watermarked image:')
ft2 = decoder(default_transform(gen_w_img).unsqueeze(0).to(device))   ###decoder的预测结果 为一个个 浮点数
decoded_msg2 = ft2 > 0 # b k -> b k      ##根据正负值 分为0-1字符串
accs2 = (~torch.logical_xor(decoded_msg2, msg_ori)) # b k -> b k
print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")
