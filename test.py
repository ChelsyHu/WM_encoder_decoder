import argparse
import datetime
import json
import os
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

import data_augmentation
import utils
import utils_img
import models7 as models 
import attenuations

from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from diffusers import  AutoencoderTiny  ##vae的精简版
from DeepCache import DeepCacheSDHelper ##用于加速diffusion过程

import argparse
import json
import os
import shutil
import tqdm
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import utils
import utils_img
import utils_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import sys
sys.path.append(os.getcwd()) 

import models7 as models 

## 1. used model to watermark images in testing dataset  and save the watermarked images 
#   (image names cannot change from original ones)
## 2. extracting the watermark based on the watermarked_images, saving the watermarks extracted by the model
#  3. determine whether watermark is extracted successfully.

## 4. generate images based on images and watermarked_images, 
# determine whether the quality of images has declined with the processing of our model. ()
## test: NCC FID Clip-score between generated images and original images
## test: PSNR SSIM between adversial images and origianl images
# 每张图片分开存，之后再找一些合在一起成为网格图片
## 5. attack (attack the watermarked_images, and then,extract watermark)


##start now!
def save_imgs(img_dir, img_dir_nw, save_dir, num_imgs=None, mult=10):
    filenames = os.listdir(img_dir)
    filenames.sort()
    if num_imgs is not None:
        filenames = filenames[:num_imgs]
    for ii, filename in enumerate(tqdm.tqdm(filenames)):
        img_1 = Image.open(os.path.join(img_dir_nw, filename))
        img_2 = Image.open(os.path.join(img_dir, filename))
        diff = np.abs(np.asarray(img_1).astype(int) - np.asarray(img_2).astype(int)) *10
        diff = Image.fromarray(diff.astype(np.uint8))
        shutil.copy(os.path.join(img_dir_nw, filename), os.path.join(save_dir, f"{ii:02d}_nw.png"))
        shutil.copy(os.path.join(img_dir, filename), os.path.join(save_dir, f"{ii:02d}_w.png"))
        diff.save(os.path.join(save_dir, f"{ii:02d}_diff.png"))


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa('--pretrained_model_name_or_path', type=str, default= "/hhd2/hqq/stable_signature/stable-diffusion-2-1-base")
    aa("--train_dir", type=str, default="datasets/train_dir")
    aa("--val_dir", type=str, default="datasets/val_dir")
    aa("--test_dir", type=str, default="datasets/Test_specific_WikiArt3")
    aa("--output_dir", type=str, default="output/output_bit/5", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')  ##fucus   观察是那里了调用了它们
    aa("--num_bits", type=int, default=48, help="Number of bits of the watermark (Default: 32)")  ###比特串长度
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=256, help="Image size")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="hidden", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")
    

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=1, type=int)
    aa('--saveckp_freq', default=100, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=0.3, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")


    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=16, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=1, help="Number of workers for data loading. (Default: 8)")   ####

    group = parser.add_argument_group('Attenuation parameters')  ##衰减参数
    aa("--attenuation", type=str, default='jnd', help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=False, help="Use channel scaling. (Default: True)")



    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="combined", help="Type of data augmentation to use at marking time. (Default: combined)")
    aa("--p_crop", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_res", type=float, default=0.5, help="Probability of the res augmentation. (Default: 0.5)")
    aa("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    aa("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)   ##important in the training with muti-GPUs,  置为0 当多GPU的时候
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=False, help='Enabling distributed training')  ##用多GPU的时候  需要置为True

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')

    return parser


def message_loss(fts, targets, m, loss_type='mse'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')



def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]



def main(params):

    # Set seeds for reproductibility 
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    os.makedirs(os.path.join(params.output_dir,'test'), exist_ok=True   )
    os.makedirs(os.path.join(params.output_dir,'test','original_imgs'), exist_ok=True   )
    os.makedirs(os.path.join(params.output_dir,'test','watermarked_imgs'), exist_ok=True   )
    
    os.makedirs(os.path.join(params.output_dir,'test','generate_imgs'), exist_ok=True   )
    os.makedirs(os.path.join(params.output_dir,'test','generate_w_imgs'), exist_ok=True   )
    

    prompts=[ "A modern reinterpretation of this image in the style of Van Gogh.",
    "Transform this image into a watercolor painting.",
    "Reimagine this scene in a cyberpunk aesthetic.",
    "Convert this image to look like a vintage photograph from the 1920s.",

    "Enhance the details and textures in this image to make it look more realistic.",
    "Increase the resolution of this image while preserving its original style.",
    "Sharpen the edges and add depth to this image.",

    "Give this image a mysterious and eerie atmosphere.",
    "Make this image feel more joyful and vibrant.",
    "Add a sense of tranquility and calm to this scene.",

    "Create an impressionist version of this image.",
    "Turn this image into a pop art masterpiece.",
    "Apply a cubist styl to this image.",

    "Change the color palette of this image to sepia tones.",
    "Make this image monochromatic with a focus on blue shades.",
    "Convert this image to a high-contrast black and white.",

    "Enlarge this image while maintaining its aspect ratio.",
    "Crop this image to focus on the main subject.",
    "Change the perspective of this image to a bird's-eye view.",

    "Reimagine this image as if it were taken in a futuristic city with neon lights and flying cars.",
    "Transform this landscape into a fantasy world with mythical creatures and magical elements.",
    "Combine the style of this image with elements from a famous painting."                 
    ]

        # Data loaders
    

    # Build encoder
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'vit':
        encoder = models.VitEncoder(
            img_size=params.img_size, patch_size=16, init_values=None,
            embed_dim=params.encoder_channels, depth=params.encoder_depth, 
            num_bits=params.num_bits, last_tanh=params.use_tanh
            )
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    else:
        raise ValueError('Unknown decoder type')
    print('\ndecoder: \n%s'% decoder)
    print('total parameters: %d'%sum(p.numel() for p in decoder.parameters()))
    
    # Adapt bn momentum
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None

    # Construct attenuation  衰减
    if params.attenuation == 'jnd':
        attenuation = attenuations.JND(preprocess = lambda x: utils_img.unnormalize_rgb(x)).to(device)
    else:
        attenuation = None

    # Construct data augmentation seen at train time
    if params.data_augmentation == 'combined':
        data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
    elif params.data_augmentation == 'kornia':
        data_aug = data_augmentation.KorniaAug().to(device)
    elif params.data_augmentation == 'none':
        data_aug = nn.Identity().to(device)
    else:
        raise ValueError('Unknown data augmentation type')
    print('data augmentation: %s'%data_aug)
    
    # Create encoder/decoder
    encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
        params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)   ##################focus
    encoder_decoder = encoder_decoder.to(device).eval()

    encoder_with_jnd = models.EncoderWithJND(
        encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
    )
    encoder_with_jnd = encoder_with_jnd.to(device).eval()
    decoder = decoder.to(device).eval()

    # 以下代码引入SDXL模型
    model= params.pretrained_model_name_or_path
    pipeline_image2image = AutoPipelineForImage2Image.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    # Change the memory layout.
    pipeline_image2image.unet.to(memory_format=torch.channels_last)
    pipeline_image2image.vae.to(memory_format=torch.channels_last)

    pipeline_image2image.unet.requires_grad_(False)
    pipeline_image2image.vae.requires_grad_(False)
    pipeline_image2image.text_encoder.requires_grad_(False)

    pipeline_image2image.set_progress_bar_config(disable=True)
    pipeline_image2image.enable_xformers_memory_efficient_attention()
    pipeline_image2image.disable_attention_slicing()

    ##加速GPU
    helper = DeepCacheSDHelper(pipe=pipeline_image2image)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()


    # test_transform = transforms.Compose([
    #     transforms.Resize(params.img_size),
    #     transforms.CenterCrop(params.img_size),
    #     transforms.ToTensor(),
    #     utils_img.normalize_rgb,
    #     ]) 
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    test_transform = transforms.Compose([
        transforms.Resize((params.img_size, params.img_size), Image.BICUBIC),
        transforms.ToTensor(), 
        NORMALIZE_IMAGENET
    ])
    test_loader = utils.get_dataloader(params.test_dir, transform=test_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)

    print('Testing...')
    start_time = time.time()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Test: '

    output_folder1 = os.path.join(params.output_dir,'test','original_imgs')
    output_folder2 = os.path.join(params.output_dir,'test','watermarked_imgs')

    output_folder3 = os.path.join(params.output_dir,'test','generate_imgs')
    output_folder4 = os.path.join(params.output_dir,'test','generate_w_imgs') 

    with torch.no_grad():
        for it, (imgs, _) in enumerate(metric_logger.log_every(test_loader, 10,header)):
            imgs = imgs.to(device, non_blocking=True) # b c h w
            #latents_o=latents_cache2[it*params.batch_size_eval: (it+1)*params.batch_size_eval].to(device)

            # msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
            msgs = '100101101011101100100110010100001101011011100110'
            msg_ori = torch.Tensor(str2msg('100101101011101100100110010100001101011011100110')).unsqueeze(0)
            msgs_ori =msg_ori.repeat(imgs.shape[0],1)

            msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

            # fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, eval_mode=True)
            imgs_w = encoder_with_jnd(imgs, msgs)

            ## 对原图片进行归一化 约束到0-1之间
            clip_imgs = torch.clamp(UNNORMALIZE_IMAGENET(imgs), 0, 1)
            clip_imgs = torch.round(255 * clip_imgs)/255 
            imgs =clip_imgs
            ## 对水印图片进行归一化 0-1之间
            clip_imgs = torch.clamp(UNNORMALIZE_IMAGENET(imgs_w), 0, 1)
            clip_imgs = torch.round(255 * clip_imgs)/255 
            imgs_w =clip_imgs
            
            # 保存生成的图像
            for i, image in enumerate(imgs):
                save_image(image, f"{output_folder1}/{it*params.batch_size_eval + i}.png" )
        
            # 保存生成的图像
            for i, image in enumerate(imgs_w):
                save_image(image, f"{output_folder2}/{it*params.batch_size_eval + i}.png" )
            

            ## generation stats

            for prompt in prompts:
                prompt1=[prompt]*imgs.shape[0]
                imgs_generate =  pipeline_image2image(prompt1, image=imgs, strength=0.5, guidance_scale=3.0,output_type='pt').images 
                imgs_w_generate= pipeline_image2image(prompt1, image=imgs_w, strength=0.5, guidance_scale=3.0, output_type='pt').images 

                # prompt_folder = f"{output_folder3}/{prompt.replace(' ', '_')}"
                # if not os.path.exists(prompt_folder):
                #     os.makedirs(prompt_folder) 
                # prompt_folder = f"{output_folder4}/{prompt.replace(' ', '_')}"
                # if not os.path.exists(prompt_folder):
                #     os.makedirs(prompt_folder)

                for i, image in enumerate(imgs_generate):
                    save_image(image, f"{output_folder3}/{prompt.replace(' ', '_')}_{it*params.batch_size_eval + i}.png" )
                for i, image in enumerate(imgs_w_generate):
                    save_image(image, f"{output_folder4}/{prompt.replace(' ', '_')}_{it*params.batch_size_eval + i}.png" )
                
                torch.cuda.empty_cache()

            # log_stats = {
            #     'loss_w': loss_w.item(),
            #     'loss_i': loss_i.item(),
            #     #'loss_latents':loss_latents.item(), 
            #     'loss': loss.item(),
            #     'psnr_avg': torch.mean(psnrs).item(),
            #     'ssim': ssim,
            #     'bit_acc_avg': torch.mean(bit_accs).item(),
            #     'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            #     'norm_avg': torch.mean(norm).item(),
            # }


            # attacks = {
            #     'none': lambda x: x, #
            #     'crop_01': lambda x: utils_img.center_crop(x, 0.1), #
            #     'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            #     #'resize_03': lambda x: utils_img.resize(x, 0.3),
            #     'resize_05': lambda x: utils_img.resize(x, 0.5),
            #     'resize_07': lambda x: utils_img.resize(x, 0.7),  # 
            #     'rot_25': lambda x: utils_img.rotate(x, 25),
            #     'rot_90': lambda x: utils_img.rotate(x, 90),
            #     'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
            #     'brightness_2': lambda x: utils_img.adjust_brightness(x, 2), #
            #     'contrast_2': lambda x: utils_img.adjust_contrast(x, 2.0) ,
            #     'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            # }
            # for name, attack in attacks.items():
            #     fts, (_) = encoder_decoder(imgs, msgs, eval_mode=True, eval_aug=attack)
            #     decoded_msgs = torch.sign(fts) > 0 # b k -> b k
            #     diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
            #     log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

            # torch.cuda.synchronize()
            # for name, loss in log_stats.items():
            #     metric_logger.update(**{name:loss})
            
            # if  it == 0 and utils.is_main_process():
            #     save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, 'test', f'{it:03}_val_ori.png'), nrow=8)
            #     save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, 'test',f'{it:03}_val_w.png'), nrow=8)

        # metric_logger.synchronize_between_processes()
        # #print("Averaged {} stats:".format('test'), metric_logger)
        # test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # log_stats = {**log_stats, **{f'test_{k}': v for k, v in test_stats.items()}}

        # if utils.is_main_process():
        #     with (Path(params.output_dir) /'test'/ "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
