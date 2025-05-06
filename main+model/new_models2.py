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
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)    # b c h w  -> b channels_out h w 





class Encoder(nn.Module):
    """
    Inserts the watermark into the image

    if the watermark is QR-CODE, channel_of_watermark=1, 
    if the watermark is also an 3-channels image, channel_of_watermark=3
    """
    def __init__(self, num_blocks, channel_of_watermark, channels, last_tanh=True ):
        super(Encoder, self).__init__()
        layers=[ConvBNRelu(4, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)
        
        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + channel_of_watermark, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.Relu = nn.ReLU(True)
        

    def forward(self, image, watermark):
        """
        image:  b 3 h w,   watermark: b*1*h*w
        output: b 3 h w 
        """
        img= torch.cat([image, watermark], dim=1 )
        encoded_image=self.conv_bns(img)  # b c h w 

        concat = torch.cat( [watermark ,encoded_image, image], dim=1)  # b (channel_of_watermark+c+3) h w 
        im_w = self.after_concat_layer(concat)  #  b c h w 
        im_w = self.final_layer(im_w)   # b 3 h w 

        im_w = self.Relu(im_w)  # b 3 h w 
        
        return im_w 
    

class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.        
    """
    def __init__(self, num_blocks, channel_of_watermark, channels):

        super(Decoder, self).__init__()

        # layers1 = [ConvBNRelu(3, channels)]
        # for _ in range(num_blocks - 1):
        #     layers1.append(ConvBNRelu(channels, channels))

        # layers1.append(ConvBNRelu(channels, channel_of_watermark))  # b*channel_of_watermark*h*w
        # # layers.append(nn.AdaptiveAvgPool2d(output_size=(h,w)))   #不知道需不需要池化层
                
        # self.layers1 = nn.Sequential(*layers1)
        # # self.linear = nn.Linear(num_bits, num_bits)


        layers1=[]
        layers1.append( nn.Conv2d(3, 3*2, kernel_size=4, stride=2, padding=1)  )  # b 4 128 128  ->  b c 64 64 
        layers1.append( nn.BatchNorm2d(3*2) )  
        layers1.append( nn.ReLU(True)  )

        layers1.append( nn.Conv2d( 3*2, 3*4, kernel_size=4, stride=2, padding=1)  )  # b c 64 64   ->  b c 32 32 
        layers1.append( nn.BatchNorm2d(3*4) )  
        layers1.append( nn.ReLU(True)  )

        # layers1.append(  nn.Conv2d( 3*4, 3*8, kernel_size=4, stride=2, padding=1)  )  # b c 32 32    ->  b c 16 16 
        # layers1.append( nn.BatchNorm2d( 3*8 ) )  
        # layers1.append( nn.ReLU(True)  )


        self.layers1 = nn.Sequential(*layers1)



        layers2= []
        # layers2.append( nn.ConvTranspose2d(3*8, 3*4 ,kernel_size=4,stride=2,padding=1  ) ) ## b c 16 16  ->
        # layers2.append(nn.BatchNorm2d( 3*4 ) )   
        # layers2.append( nn.ReLU(True))

        layers2.append( nn.ConvTranspose2d(3*4,3*2,kernel_size=4,stride=2,padding=1  ) )
        layers2.append(nn.BatchNorm2d( 3*2 ) )
        layers2.append( nn.ReLU(True))

        layers2.append( nn.ConvTranspose2d(3*2,1,kernel_size=4,stride=2,padding=1  ) ) #  b c 64 64 -> b 3 128 128 
        layers2.append(nn.BatchNorm2d( 1 ) )
        layers2.append( nn.ReLU(True))

        self.layers2 = nn.Sequential(*layers2)

        

    def forward(self, img_w):

        x = self.layers1(img_w) # b channel_of_watermark h w 
        # x = x.squeeze(-1).squeeze(-1) # b d
        # x = self.linear(x) # b d
        x=self.layers2(x)>0  # b channel_of_watermark h w 
        # x=nn.Sigmoid(x)
        x=x+0.0
        return x   
    



class Decoder_Generator(nn.Module):
    def __init__(self,
                 gen_input_nc=3,  ## original image channel  
                 image_nc=1,  ## watermark channel
                 ):
        super(Decoder_Generator, self).__init__()

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(True),
            # 8*64*64
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            # 16*32*32
        ]

        bottle_neck_lis = [ResnetBlock(16),
                       ResnetBlock(16),
                       ResnetBlock(16),
                       ResnetBlock(16),]

        decoder_lis = [
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(True),
            # state size. 8 * 64 * 64
            nn.ConvTranspose2d(8, image_nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True)  # 输出在0~1     
            # state size.  1 * 128 * 128 
        ]  

        self.en = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.de= nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.en(x) 
        x = self.bottle_neck(x)
        x = self.de(x)
        x=(x>0)+0.0
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class EncoderDecoder(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        attenuation: attenuations.JND, 
        augmentation: nn.Module,  ###
        decoder:nn.Module,
        scale_channels: bool,
        scaling_i: float,
        scaling_w: float,
        channel_of_watermark: int,
        redundancy: int
    ):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation  ##
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.channel_of_watermark = channel_of_watermark
        self.redundancy = redundancy

    def forward(
        self,
        imgs: torch.Tensor,
        watermark: torch.Tensor,
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
            watermark: b 1 h w,  或者 b 3 h w 
        """

        # encoder
        deltas_w = self.encoder(imgs, watermark)  # b 3 h w

        # # scaling channels: more weight to blue channel
        # if self.scale_channels:
        #     aa = 1/4.6 # such that aas has mean 1
        #     aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device) 
        #     deltas_w = deltas_w * aas[None,:,None,None]

        # # add heatmaps   ps: 调参的时候可能还需要focus
        # if self.attenuation is not None:
        #     heatmaps = self.attenuation.heatmaps(imgs) # b 1 h w
        #     deltas_w = deltas_w * heatmaps # # b c h w * b 1 h w -> b c h w
        
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w # b c h w

        ##此处的img_w就是我们打好了水印，并且调节imgs, deltas_w比例之后的图片 ，  是用户发布到网络上面的图片

        # data augmentation
        if eval_mode:     ## evaluation mode 
            #imgs_aug = eval_aug(imgs_w)
            #imgs_aug=imgs_aug.resize( imgs_w.shape[0], imgs_w.shape[1], imgs_w.shape[-2],imgs_w.shape[-1] )
            #wm2 = self.decoder(imgs_aug) # b c h w -> b channel_of_watermark h w 

            # img_c 是concat 图片和watermark
            imgs_aug = eval_aug(imgs_w)
            # img_c= torch.cat([imgs_aug, watermark], dim=1 )  ## b c+1 h w 
            wm2= self.decoder(imgs_aug) 

        else:  ## training mode 
            #imgs_aug = self.augmentation(imgs_w)   ##这一步，把128的size 换成了70
            #wm2 = self.decoder(imgs_aug) # b c h w -> b channel_of_watermark h w 
            #img_c= torch.cat([imgs_w, watermark], dim=1 )
            wm2 = self.decoder(imgs_w)
        # fts = fts.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        # fts = torch.sum(fts, dim=-1) # b k r -> b k

        return wm2, imgs_w  #, imgs_aug