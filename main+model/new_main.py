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
import new_models
import attenuations
import qrcode 
import random 

import os 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import sys 
sys.path.append(os.getcwd()) 
sys.path.append('src')

from src.loss.ssim import ssim
from src.loss.watson import WatsonDistance
from src.loss.watson_vgg import WatsonDistanceVgg
from src.loss.watson_fft import WatsonDistanceFft
from src.loss.shift_wrapper import ShiftWrapper
from src.loss.deep_loss import PNetLin
from PIL import Image



def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="datasets/train_dir")
    aa("--val_dir", type=str, default="datasets/val_dir")
    aa("--output_dir", type=str, default="output/ssim", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')  ##fucus   观察是那里了调用了它们
    #aa("--num_bits", type=int, default=32, help="Number of bits of the watermark (Default: 32)")  ###比特串长度
    aa("--channel_of_watermark", type=int, default=1,help='channel of the watermark(Default:1)' )
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=128, help="Image size")
    aa("--WM_info", type=str, default='Michelangelo', help="Information that you want to insert to the watermark" )  ##水印包含的信息

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="hidden", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")
    # encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    
    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="Decoder", help="Decoder type (Default: Decoder. Choices:Decoder, Decoder_Generator)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.5, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=1, type=int)
    aa('--saveckp_freq', default=100, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=100, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--lambda_w", type=float, default=50, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1, help="Weight of the image loss. (Default: 0.0)")
    #aa("--lambda_white",type=float, default=0, help="Weight of the white edge loss. (Default: 1.0) ")
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='huber', help="Loss type.'ssim', 'mse' for mean squared error, 'l1' for l1 loss, 'WatsonDistance', 'WatsonDistanceVgg','WatsonDistanceFft','ShiftWrapper','PNetLin' (Default: mse)")
    
    """ssim', 'mse' for mean squared error, 'l1' for l1 loss, 
    "WatsonDistance", "WatsonDistanceVgg","WatsonDistanceFft","ShiftWrapper","PNetLin"
    """
    
    aa("--loss_w_type", type=str, default='ssim', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")
    aa("--lr",  type=float, default= 1e-3, help="Learning rate (Default: 1e-3)"  )

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=4, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=4, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=2, help="Number of workers for data loading. (Default: 8)")

    group = parser.add_argument_group('Attenuation parameters')  ##衰减参数
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="combined", help="Type of data augmentation to use at marking time. (Default: combined)")
    aa("--p_crop", type=float, default=0, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_res", type=float, default=0.5, help="Probability of the res augmentation. (Default: 0.5)")
    aa("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    aa("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)   ## important in the training with muti-GPUs,  置为0 当多GPU的时候,否则为-1
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=False, help='Enabling distributed training')  ## 用多GPU的时候  需要置为Truet

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')

    return parser



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
    # if params.encoder == 'hidden':
    #     encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    # elif params.encoder == 'dvmark':
    #     encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    # elif params.encoder == 'vit':
    #     encoder = models.VitEncoder(
    #         img_size=params.img_size, patch_size=16, init_values=None,
    #         embed_dim=params.encoder_channels, depth=params.encoder_depth, 
    #         num_bits=params.num_bits, last_tanh=params.use_tanh
    #         )
    # else:
    #     raise ValueError('Unknown encoder type')

    encoder=new_models.Encoder(num_blocks=params.encoder_depth, channel_of_watermark=params.channel_of_watermark, channels=params.encoder_channels, last_tanh=params.use_tanh)
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    # if params.decoder == 'hidden':
    #     decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    # else:
    #     raise ValueError('Unknown decoder type')
    
    if params.decoder == "Decoder":
        decoder = new_models.Decoder(num_blocks=params.decoder_depth, channel_of_watermark=params.channel_of_watermark, channels=params.decoder_channels)
    if params.decoder == 'Decoder_Generator':
        decoder = new_models.Decoder_Generator()
    
    print('\ndecoder: \n%s'% decoder)
    print('total parameters: %d'%sum(p.numel() for p in decoder.parameters()))
    
    # Adapt bn momentum
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None

    # Construct attenuation
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
    encoder_decoder = new_models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
        params.scale_channels, params.scaling_i, params.scaling_w, params.channel_of_watermark, params.redundancy)   ##################focus
    encoder_decoder = encoder_decoder.to(device)


    # Distributed training
    if params.dist: 
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
        encoder_decoder = nn.parallel.DistributedDataParallel(encoder_decoder, device_ids=[params.local_rank])

    # Build optimizer and scheduler
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
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

    # optionally resume training 
    if params.resume_from is not None: 
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder_decoder=encoder_decoder,
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

        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if epoch % params.eval_freq == 0:
            val_stats = eval_one_epoch(encoder_decoder, val_loader, epoch, params)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
    
        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
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




def message_loss(wm1, wm2, m, loss_type='mse'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    # if loss_type == 'bce':
    #     return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    # elif loss_type == 'cossim':
    #     return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    # elif loss_type == 'mse':
    return F.mse_loss(wm1, wm2, reduction='mean')
    # else:
    #     raise ValueError('Unknown loss type')



def image_loss(imgs, imgs_ori, loss_type='huber'):
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
    if loss_type =='huber':
        return F.huber_loss(imgs, imgs_ori)
    if loss_type == 'ssim':
        return ssim(imgs, imgs_ori)
       #SSIM(size_average=(reduction in ['sum', 'mean']))
    if loss_type == 'WatsonDistance':
        return WatsonDistance( imgs, imgs_ori)
    if loss_type == 'WatsonDistanceVgg':
        return WatsonDistanceVgg(imgs, imgs_ori)
    if loss_type == "WatsonDistanceFft":
        return WatsonDistanceFft(imgs, imgs_ori)
    if loss_type == "ShiftWrapper":
        return ShiftWrapper(imgs, imgs_ori)
    if loss_type == "PNetLin":
        return PNetLin(imgs, imgs_ori)
    #'WatsonDistance', 'WatsonDistanceVgg','WatsonDistanceFft','ShiftWrapper','PNetLin'
    else:
        raise ValueError('Unknown loss type')

def white_edge_loss(watermark, wm2 ):
            ## constraining the fixed white edge
    zero_indices = np.argwhere(watermark == 0.0) ##         print("零元素的位置：")
    min1= min(zero_indices[1])
    max1= max(zero_indices[1])
    h, w = wm2.shape[-2],wm2.shape[-1] 
    loss_white= 0 

    for number in range(wm2.shape[0]):
        for i in range(0,min1):
            loss_white+= wm2[number][0][i].sum()
        for i in range( max1+1, h):
            loss_white+= wm2[number][0][i].sum()

    wm2=wm2.permute(0, 1, 3, 2)
    h, w = wm2.shape[-2],wm2.shape[-1]
    for number in range(wm2.shape[0]):
        for i in range(0,min1):
            loss_white+= wm2[number][0][i].sum()
        for i in range( max1+1, h):
            loss_white+= wm2[number][0][i].sum()

    return loss_white 



def train_one_epoch(encoder_decoder: new_models.EncoderDecoder, loader, optimizer, scheduler, epoch, params):
    """
    One epoch of training.
    """
    if params.scheduler is not None:
        scheduler.step(epoch)
    encoder_decoder.train() 
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        artist_name=params.WM_info
        # 设置词组长度
        img_count = imgs.shape[0]


        # random_words = [artist_name for ii in range(word_count)]
        # # #print(random_words)

        # for i in range(word_count):
        #     wm= qrcode.make(random_words[i], box_size=1,version=1,error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
        #     wm =wm.resize([imgs.shape[-2],imgs.shape[-1]])
        #     transform = transforms.ToTensor()
        #     watermark = transform(wm)
        #     watermark = watermark.unsqueeze(0)  # 1*h*w 
        #     if i==0:
        #         watermark_b=watermark
        #     else:
        #         watermark_b=np.vstack( [watermark_b, watermark] )  ## b * 1 * h *w 

        # watermark_b=torch.from_numpy(watermark_b).to(device,non_blocking=True)

        # 打开图片文件
        image1 = Image.open('imgs/QR_code/4.png')
        # 定义转换
        transform = transforms.ToTensor()
        # 转换图像为张量
        tensor_image = transform(image1)
        watermark = tensor_image.unsqueeze(0)   # 1 1 h w 
        count = imgs.shape[0]

        for i in range(count):
            if i == 0:
                watermark_b=watermark 
            else:
                watermark_b=np.vstack([watermark_b,watermark])
                
        watermark_b=torch.from_numpy(watermark_b).to(device,non_blocking=True) ## b 1 h w 

        wm2, imgs_w = encoder_decoder(imgs, watermark_b)  ##, imgs_aug 可能输出
        #wm2 b 1 70 70,   imgs_aug b  c 70 70 ,  imgs_w  b c h w,  
        watermark_b=watermark_b.float()
        wm2=wm2.float()

        # wm2 = torch.sign(wm2) > 0 # b k -> b k
        loss_w = image_loss( wm2, watermark_b, loss_type=params.loss_i_type) # b 1 h w -> 1
        loss_i = image_loss( imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1
        # loss_white = white_edge_loss( watermark, wm2 ) 
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i  #+ params.lambda_white*loss_white  ## 

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1
        # msg stats
        # ori_msgs = torch.sign(msgs) > 0
        # decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        # diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        # bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        # word_accs = (bit_accs == 1) # b
        # norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1

        # QR stats
        ori_wm=watermark_b
        decoded_wm = wm2 


        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            #'loss_white_edge': loss_white.item(),
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
            # save_image(utils_img.unnormalize_img(imgs_aug), os.path.join(params.output_dir,'train', f'{epoch:03}_{it:03}_train_aug.png'), nrow=8)
            save_image(watermark_b, os.path.join(params.output_dir, 'train',f'{epoch:03}_{it:03}_original_WM.png'), nrow=8)
            save_image(wm2, os.path.join(params.output_dir, 'train',f'{epoch:03}_{it:03}_decoded_WM.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def eval_one_epoch(encoder_decoder: new_models.EncoderDecoder, loader, epoch, params):
    """
    One epoch of eval.
    """
    encoder_decoder.eval()
    header = 'Eval - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True) # b c h w

        # artist_name=['Michelangelo',"Vincent van Gogh","Caravaggio",
        # "Rembrandt","Leonardo da Vinci","Johannes Vermeer",
        # "Claude Monet","Raphael","Pablo Picasso","Diego Velázquez",
        # "Salvador Dalí","Pierre-Auguste Renoir","Peter Paul Rubens",
        # "Édouard Manet", "Paul Cézanne", "Edgar Degas",
        # "Gustav Klimt", "Edvard Munch", "Henri Matisse","Georges-Pierre Seurat"
        # ]
        # # 设置词组长度
        # word_count = imgs.shape[0]

        artist_name=params.WM_info
        # 设置词组长度
        word_count = imgs.shape[0]
        random_words = [artist_name for ii in range(word_count)]

        for i in range(word_count):
            wm= qrcode.make(random_words[i], box_size=10,version=1,error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
            wm =wm.resize([imgs.shape[-2],imgs.shape[-1]])
            transform = transforms.ToTensor()
            watermark = transform(wm)
            watermark = watermark.unsqueeze(0)  # 1*h*w 
            if i==0:
                watermark_b=watermark
            else:
                watermark_b=np.vstack( [watermark_b, watermark] )  ## b * 1 * h *w 

        watermark_b=torch.from_numpy(watermark_b).to(device,non_blocking=True)

        #print(watermark_b.shape)
        wm2, imgs_w = encoder_decoder(imgs, watermark_b, eval_mode=True)    ## imgs_aug
        
        watermark_b=watermark_b.float()
        wm2=wm2.float()
        #print(wm2.shape, imgs_w)
        # msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        # msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

        # fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, eval_mode=True)

        loss_w = image_loss( wm2, watermark_b,  loss_type=params.loss_w_type) # b 1 h w -> 1
        loss_i = image_loss(imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1
        #loss_white = white_edge_loss( watermark, wm2 ) 
        loss = params.lambda_w*loss_w + params.lambda_i*loss_i    #+ params.lambda_white*loss_white  ## 

        # img stats
        psnrs = utils_img.psnr(imgs_w, imgs) # b 1
        # # msg stats
        # ori_msgs = torch.sign(msgs) > 0
        # decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        # diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        # bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        # word_accs = (bit_accs == 1) # b
        # norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
        # log_stats = {
        #     'loss_w': loss_w.item(),
        #     'loss_i': loss_i.item(),
        #     'loss': loss.item(),
        #     'psnr_avg': torch.mean(psnrs).item(),
        #     'bit_acc_avg': torch.mean(bit_accs).item(),
        #     'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
        #     'norm_avg': torch.mean(norm).item(),
        # }

        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            #'loss_white':loss_white.item(),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
        }        
        
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            # 'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
            # 'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }

        ## focus:
        # for name, attack in attacks.items():
        #     wm2 , (_) = encoder_decoder(imgs, watermark_b, eval_mode=True, eval_aug=attack)
        #     wm2 = torch.sign(wm2) > 0 # b k -> b k
        #     diff = (~torch.logical_xor(watermark_b, wm2)) # b k -> b k
        #     log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        
        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(utils_img.unnormalize_img(imgs), os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(utils_img.unnormalize_img(imgs_w), os.path.join(params.output_dir,'validation', f'{epoch:03}_{it:03}_val_w.png'), nrow=8)


            save_image(watermark_b, os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_original_WM.png'), nrow=8)
            save_image(wm2, os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_decoded_WM.png'), nrow=8)

            # save_image(utils_img.QR_img(watermark_b), os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_original_WM.png'), nrow=8)
            # save_image(utils_img.QR_img(wm2), os.path.join(params.output_dir, 'validation',f'{epoch:03}_{it:03}_decoded_WM.png'), nrow=8)
            
            

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)