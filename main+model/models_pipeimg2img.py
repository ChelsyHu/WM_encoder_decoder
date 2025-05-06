import torch
import torch.nn as nn
from timm.models import vision_transformer

import attenuations
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers import AutoPipelineForImage2Image

def img2img(imgs_tensor,prompts, pipeline):  
    torch.set_grad_enabled(True)
    #image = pipeline_image2image(prompt, image=img, strength=0.8, guidance_scale=10.5).images #
    image = pipeline(prompts, image=imgs_tensor, strength=0.8, guidance_scale=10.5,output_type='np').images # 
    # 将 NumPy 数组转换回 PyTorch 张量，并确保 requires_grad=True
    gen_image = torch.from_numpy(image).permute(0,3,1,2).float().requires_grad_(True)

    del pipeline
    # 假设pipe是一个大型的模型或数据加载器
    torch.cuda.empty_cache()  # 清理未使用的GPU内存

    return gen_image



def latent2latent(img_latents_w,noise_scheduler,unet, text_embeddings):

    latents = img_latents_w * noise_scheduler.init_noise_sigma

    text_embeddings=text_embeddings.repeat(latents.shape[0],1,1)
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
                    # Sample a random timestep for each image
    timesteps = torch.randint(
                        0,
                        25,
                        #noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
    timesteps = timesteps.long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    unet.requires_grad_(False)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample  # [8 4 64 64],  [2,77,1024]


    model_pred=1/(0.18215) *model_pred 
    # ## decode to get the final image 
    # img_pred= vae.decode(model_pred).sample

    return model_pred

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

        # self.last_tanh = last_tanh
        # self.tanh = nn.Tanh()
        self.sigmod=nn.Sigmoid()
        

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs) # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1) # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)  # b 3 h w 

        # if self.last_tanh:
        #     im_w = self.tanh(im_w)
        im_w = self.sigmod(im_w)

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
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        x= torch.tanh(x)

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
        redundancy: int,
        #device,
        pipeline_image2image: AutoPipelineForImage2Image,
        prompts:list
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
        #self.device=device 
        self.pipeline_image2image=pipeline_image2image
        self.prompts=prompts
        # #self.vae=vae
        # self.noise_scheduler=noise_scheduler
        # self.unet=unet
        # self.text_embeddings=text_embeddings

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
        deltas_w = self.encoder(imgs, msgs) # b 3 h w 

        imgs_w = (self.scaling_i * imgs + self.scaling_w * deltas_w)/(self.scaling_i+self.scaling_w) # b c h w

        imgs_w_generation = img2img(imgs_w,  self.prompts, self.pipeline_image2image  )

        #imgs_w_generation=latent2latent(imgs_w,self.noise_scheduler,self.unet, self.text_embeddings)
        #fts = self.decoder(imgs_w_generation)  # b c h w -> b d
        fts = self.decoder(imgs_w)  # b c h w -> b d

        fts = fts.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        fts = torch.sum(fts, dim=-1) # b k r -> b k

        # del self.pipeline_image2image

        # return fts, (imgs_w, imgs_aug)
        #return fts, (imgs_generation, imgs_w, imgs_w_generation)
        #return fts, (imgs_w, imgs_w_generation)
        return fts, (imgs_w, imgs_w_generation)

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