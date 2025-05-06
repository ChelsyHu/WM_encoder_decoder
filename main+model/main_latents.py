"""
torchrun --nproc_per_node=2 main.py \
    --local_rank 0 \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100 --scaling_w 0.5 \
    --batch_size 16 --eval_freq 10 \
    --attenuation jnd \
    --epochs 100 --optimizer "AdamW,lr=1e-4"
    
Args Inventory:
    --dist True \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --batch_size 128 --batch_size_eval 128 --workers 2 \
    --attenuation jnd \
    --num_bits 64 --redundancy 16 \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --loss_margin 100   --attenuation jnd --batch_size 16 --eval_freq 10 --local_rank 0 \
    --p_crop 0 --p_rot 0 --p_color_jitter 0 --p_blur 0 --p_jpeg 0 --p_res 0 \
"""

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
import models as models 
import attenuations

from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from diffusers import  AutoencoderTiny  ##vae的精简版

from DeepCache import DeepCacheSDHelper ##用于加速diffusion过程

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

os.environ["RANK"]='0'
os.environ['WORLD_SIZE']='4'
os.environ['LOCAL_RANK']='0'
os.environ['n_gpu_per_node']='4'


# # 以下代码引入SDXL模型
# model= "/hhd2/hqq/stable_signature/stable-diffusion-2-1-base"

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#          model, torch_dtype=torch.float16, variant="bp16", use_safetensors=True
# ).to("cuda")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import sys
sys.path.append(os.getcwd()) 


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa('--latents_exist', type=bool,  default=True)
    aa('--pretrained_model_name_or_path', type=str, default= "/ssd-sata1/hqq/stable_signature/stable-diffusion-2-1-base")
    aa("--train_dir", type=str, default="datasets/train_dir")
    aa("--val_dir", type=str, default="datasets/val_dir")
    aa("--test_dir", type=str, default="datasets/big_WikiArt3_train")
    aa("--output_dir", type=str, default="output/output_bit/5", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')  ##fucus   观察是那里了调用了它们
    aa("--num_bits", type=int, default=32, help="Number of bits of the watermark (Default: 32)")  ###比特串长度
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=256, help="Image size")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="latent", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=8, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")
    # encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    
    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=1, type=int)
    aa('--saveckp_freq', default=20, type=int)
    aa('--saveimg_freq', default=5, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=400, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 0.0)")
    aa("--lambda_latents", type=float, default=1.0, help="Weight of the image latents(default: 0.0 )" )
    aa("--lambda_latents_b", type=float, default=1.0, help="Weight of the image latents(default: 0.0 )" )
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='mse', help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=64, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=2, help="Number of workers for data loading. (Default: 8)")   ####

    group = parser.add_argument_group('Attenuation parameters')  ##衰减参数
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

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
    aa('--local_rank', default=0, type=int)   ##important in the training with muti-GPUs,  置为0 当多GPU的时候
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=False, help='Enabling distributed training')  ##用多GPU的时候  需要置为True
    aa('--rank', type=int, default=0)

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')
    aa('--generate_eval', default=False, type=bool, help='whether generate images and evaluate them when evaluation')

    return parser

def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model



def main(params):
    # Distributed mode
    if params.dist:
        utils.init_distributed_mode(params)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    # Set seeds for reproductibility 
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))



    # ##加速GPU
    # helper = DeepCacheSDHelper(pipe=pipeline_image2image)
    # helper.set_params(cache_interval=3, cache_branch_id=0)
    # helper.enable()


    # handle params that are "none"
    if params.attenuation is not None:
        if params.attenuation.lower() == 'none':
            params.attenuation = None
    if params.scheduler is not None:
        if params.scheduler.lower() == 'none':
            params.scheduler = None
	
        # make directory:
    #os.makedirs(params.output_dir,exist_ok=True  )
    os.makedirs(os.path.join(params.output_dir,'train'), exist_ok=True)
    os.makedirs(os.path.join(params.output_dir,'validation'), exist_ok=True   )
 
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
    elif params.encoder =='latent':
        encoder = models.LatentEncoder(num_blocks=params.encoder_depth, channels=params.encoder_channels, last_tanh=params.use_tanh) 
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))

    encoder =encoder.to(device)


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
    
    # # Create encoder/decoder
    # encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
    #     params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)   ##################focus
    # encoder_decoder = encoder_decoder.to(device)


    vae = AutoencoderTiny.from_pretrained("/hhd2/hqq/stable_signature/taesd", torch_dtype=torch.float32).to('cuda')
    vae.to(memory_format=torch.channels_last)
    vae.requires_grad_(False)

    vae.enable_xformers_memory_efficient_attention()


    # base='LDM_configs/v2-inference.yaml'
    # config_path = os.path.join(os.getcwd(), base)
    # config = OmegaConf.load(config_path)

    # ckpt_path = '/hhd2/hqq/stable_signature/stable-diffusion-2-1-base/v2-1_768-nonema-pruned.ckpt'
    # sd_model = load_model_from_config(config, ckpt_path).to(device)


    # Distributed training
    if params.dist: 
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[params.local_rank]) # device_ids=None

        vae = nn.SyncBatchNorm.convert_sync_batchnorm(vae)
        vae = nn.parallel.DistributedDataParallel(vae, device_ids=[params.local_rank]) # device_ids=None



    # # Build optimizer and scheduler
    # optim_params = utils.parse_params(params.optimizer)
    # lr_mult = params.batch_size * utils.get_world_size() / 512.0 *8 
    # optim_params['lr'] = 0.001 #lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    # to_optim = [*encoder.parameters(), *decoder.parameters()]
    # optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    # #optimizer = torch.optim.AdamW(to_optim, lr=1e-3  )
    # scheduler = None  #utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    # print('optimizer: %s'%optimizer)
    # print('scheduler: %s'%scheduler)



    # Build optimizer and scheduler
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0 *8 
    optim_params['lr'] = 0.0001 #lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3  # 0.001 -> 0.0005 
    to_optim = [*encoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    #optimizer = torch.optim.AdamW(to_optim, lr=1e-3  )
    scheduler = None  #utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s'%optimizer)
    print('scheduler: %s'%scheduler)

    # Data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.img_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_rgb,
        ]) 
    train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size, num_workers=params.workers, shuffle=True)
    val_loader = utils.get_dataloader(params.val_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)
    test_loader = utils.get_dataloader(params.test_dir, transform=val_transform,  batch_size=params.batch_size_eval, num_workers=params.workers, shuffle=False)


    # optionally resume training 
    if params.resume_from is not None: 
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder=encoder
        )
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder=encoder,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    print('training...')
    start_time = time.time()
    best_bit_acc = 0
    for epoch in range(start_epoch, params.epochs):
        
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)
        
        # if epoch == 356 or epoch == 355:
        #     test_stats = test_one_epoch(encoder_decoder, test_loader, epoch, params, vae)
        #     log_stats = {**log_stats, **{f'test_{k}': v for k, v in val_stats.items()}}
        
        train_stats = train_one_epoch(encoder, train_loader, optimizer, scheduler, epoch, params,vae)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % params.eval_freq == 0:
            val_stats = eval_one_epoch(encoder, val_loader, epoch, params, vae)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
    
        save_dict = {
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        utils.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        if params.saveckp_freq and epoch % params.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth'))
        if utils.is_main_process():
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




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


def train_one_epoch(encoder: models.LatentEncoder, loader, optimizer, scheduler, epoch, params,vae):
    """
    One epoch of training.
    """
    # if params.scheduler is not None:
    #     scheduler.step(epoch)
    encoder.train()

    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")


    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        #latents_o=latents_cache[it*params.batch_size: (it+1)*params.batch_size].to(device)
        imgs = imgs.to(device, non_blocking=True) # b c h w


        imgs_w = encoder(imgs)
        loss_i = image_loss(imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1

        # loss_latents= image_loss(vae.encode(imgs_w).latents, vae.encode(imgs).latents )
        
        # 创建一个全黑的张量
        black_tensor = torch.zeros((3, params.img_size,params.img_size), dtype=torch.float32)
        black_tensor = black_tensor.repeat(imgs.shape[0],1,1,1  ).to(device)
        black_latents = vae.encode(black_tensor).latents
        loss_latents_b = image_loss(vae.encode(imgs_w).latents, black_latents)

        # z_b =sd_model.get_first_stage_encoding(sd_model.encode_first_stage(black_tensor)).to(device)
        # z =sd_model.get_first_stage_encoding(sd_model.encode_first_stage(imgs_w)).to(device)
        # loss_latents_b = image_loss(z_b, z)

        #loss = params.lambda_w*loss_w + params.lambda_i*loss_i + (1-params.lambda_latents*loss_latents) + params.lambda_latents_b*loss_latents_b

        loss =  params.lambda_i*loss_i + params.lambda_latents_b*loss_latents_b


        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1

        log_stats = {
            'loss_i': loss_i.item(),
            #'loss_latents':loss_latents.item(), 
            'loss_latents_b': loss_latents_b.item(), 
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        # if epoch % 1 == 0 and it % 10 == 0 and utils.is_main_process():
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, 'train',f'{epoch:03}_{it:03}_train_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, 'train',f'{epoch:03}_{it:03}_train_w.png'), nrow=8)
            #save_image(utils_img.unnormalize_img(imgs_aug), os.path.join(params.output_dir, 'train',f'{epoch:03}_{it:03}_train_aug.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def eval_one_epoch(encoder:models.LatentEncoder, loader, epoch, params,vae):
    """
    One epoch of eval.
    """
    encoder.eval()
    header = 'Eval - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")


    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        imgs_w = encoder(imgs)
        
        loss_i = image_loss(imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1
        
        # 创建一个全黑的张量
        black_tensor = torch.zeros((3, params.img_size,params.img_size), dtype=torch.float32)
        black_tensor = black_tensor.repeat(imgs.shape[0],1,1,1  ).to(device)
        black_latents = vae.encode(black_tensor).latents
        loss_latents_b = image_loss(vae.encode(imgs_w).latents,black_latents)
        #loss = params.lambda_w*loss_w + params.lambda_i*loss_i + (1-params.lambda_latents*loss_latents) + params.lambda_latents_b*loss_latents_b

        # z_b =sd_model.get_first_stage_encoding(sd_model.encode_first_stage(black_tensor)).to(device)
        # z =sd_model.get_first_stage_encoding(sd_model.encode_first_stage(imgs_w)).to(device)
        # loss_latents_b = image_loss(z_b, z)

        loss =  params.lambda_i*loss_i + params.lambda_latents_b*loss_latents_b


        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1
        #ssim = utils_img.ssim(imgs_w, imgs)

        ## generation stats


        log_stats = {
            'loss_i': loss_i.item(),
            #'loss_latents':loss_latents.item(), 
            'loss_latents_b': loss_latents_b.item(), 
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
        }


        # attacks = {
        #     'none': lambda x: x, #
        #     'crop_01': lambda x: utils_img.center_crop(x, 0.1), #
        #     'crop_05': lambda x: utils_img.center_crop(x, 0.5),
        #     #'resize_03': lambda x: utils_img.resize(x, 0.3),
        #     # 'resize_05': lambda x: utils_img.resize(x, 0.5),
        #     'resize_07': lambda x: utils_img.resize(x, 0.7),  # 
        #     # 'rot_25': lambda x: utils_img.rotate(x, 25),
        #     # 'rot_90': lambda x: utils_img.rotate(x, 90),
        #     # 'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
        #     'brightness_2': lambda x: utils_img.adjust_brightness(x, 2), #
        #     'contrast_2': lambda x: utils_img.adjust_contrast(x, 2.0) ,
        #     'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        # }
        # for name, attack in attacks.items():
        #     fts, (_) = encoder(imgs, msgs, eval_mode=True, eval_aug=attack)
        #     decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        #     diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        #     log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, 'validation', f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_val_w.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def test_one_epoch(encoder_decoder: models.EncoderDecoder, loader, epoch, params,vae):
    """
    One epoch of test.
    """
    encoder_decoder.eval()
    header = 'Test - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w
        #latents_o=latents_cache2[it*params.batch_size_eval: (it+1)*params.batch_size_eval].to(device)

        msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

        fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, eval_mode=True)
        

        loss_w = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type) # b k -> 1
        loss_i = image_loss(imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1

        #loss_latents= image_loss(vae.encode(imgs_w).latents, vae.encode(imgs).latents )
        
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i #+ (1-params.lambda_latents*loss_latents)
        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1
        # msg stats
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            #'loss_latents':loss_latents.item(), 
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
        }

        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            fts, (_) = encoder_decoder(imgs, msgs, eval_mode=True, eval_aug=attack)
            decoded_msgs = torch.sign(fts) > 0 # b k -> b k
            diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
            log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, 'test', f'{epoch:03}_{it:03}_test_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir, 'test',f'{epoch:03}_{it:03}_test_w.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('test'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
