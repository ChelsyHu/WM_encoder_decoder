import os
import sys 
sys.path.append(os.getcwd())

import models 
import attenuations 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

import utils
import utils_img
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import data_augmentation

os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # you should run this notebook in the root directory of the hidden project for the following imports to work

# from models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
# from attenuations import JND

test_dir="/hhd2/hqq/stable_signature/WM_encoder_decoder/small_WikiArt3_val/noise-ckpt/noise-ckpt/50"
batch_size=8
random_msg = False
num_bits = 48
save_folder = "watermark/small_WikiArt3_val/50"
os.makedirs(save_folder, exist_ok=True)
data_augmentation_mine='none'


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]


class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float, redundancy:int):
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
        self.redundancy =redundancy


NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5,redundancy=1
)

decoder = models.HiddenDecoder(
    num_blocks=params.decoder_depth, 
    num_bits=params.num_bits, 
    channels=params.decoder_channels
)
encoder = models.HiddenEncoder(
    num_blocks=params.encoder_depth, 
    num_bits=params.num_bits, 
    channels=params.encoder_channels
)
attenuation = attenuations.JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
encoder_with_jnd = models.EncoderWithJND(
    encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
)
    # Construct data augmentation seen at train time
if data_augmentation_mine == 'combined':
    data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur,  params.p_jpeg, params.p_rot,  params.p_color_jitter, params.p_res).to(device)
elif data_augmentation_mine == 'kornia':
    data_aug = data_augmentation.KorniaAug().to(device)
elif data_augmentation_mine == 'none':
    data_aug = nn.Identity().to(device)
else:
    raise ValueError('Unknown data augmentation type')
print('data augmentation: %s'%data_aug)

#ckpt_path = "/hdd2/hqq/stable_signature/hidden/output/output_bit/6/checkpoint.pth"
ckpt_path = "ckpts/hidden_replicate.pth"
state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder_with_jnd = encoder_with_jnd.to(device).eval()
decoder = decoder.to(device).eval()

# Create encoder/decoder
encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder, 
        params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)   ##################focus
encoder_decoder = encoder_decoder.to(device)




# 自定义Dataset类
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): 列表，包含图片的完整路径。
            transform (callable, optional): 一个用于进行数据变换的可调用对象。
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.image_paths[idx]
        # 打开图片
        image = Image.open(img_path)
        # 如果定义了transform，则对图片进行变换
        if self.transform:
            image = self.transform(image)
        # 获取图片名称（不包含路径）
        img_name = img_path.split('/')[-1]
        # 返回图片张量和图片名称
        return image, img_name


# 定义归一化和反归一化的参数
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 创建一个 transforms.Compose 对象，包含所有需要的转换步骤
default_transform = transforms.Compose([
    transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    NORMALIZE_IMAGENET
])

image_paths = [os.path.join(test_dir, file_name) for file_name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, file_name))]

    # 创建自定义Dataset实例
dataset = CustomImageDataset(image_paths=image_paths, transform=default_transform)
    # 创建DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for imgs, names in dataloader:
    imgs=imgs.to(device)
    #names = names.to(device)
    # # create message
    # if random_msg:
    #     msg_ori = torch.randint(0, 2, (1, num_bits), device=device).bool() # b k
    # else:
    #     msg_ori = torch.Tensor(str2msg("100101101011101100100110010100001101011011100110")).unsqueeze(0)
    
    # msg = 2 * msg_ori.type(torch.float) - 1 # b k  ###将 0，1, 分别换成了 -1， 1
    # msg=msg.to(device)  ##这个是我加的  attention
    # msg_ori=msg_ori.to(device)
    

    imgs = imgs.to(device, non_blocking=True) # b c h w 
    msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
    msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k

    fts, (imgs_w, imgs_aug) = encoder_decoder(imgs, msgs, eval_mode=True)


    # # encode
    # imgs_w = encoder_with_jnd(images, msgs)    ### encoder_with_jnd

    for img_pixel, img_name in zip(imgs_w, names):
        save_path = os.path.join(save_folder, f"{img_name}")
        # Image.fromarray(
        #                 (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        #             ).save(save_path)
        clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_pixel), 0, 1)
        clip_img = torch.round(255 * clip_img)/255 
        Image.fromarray( clip_img).save(save_path)

    # msg stats
    ori_msgs = torch.sign(msgs) > 0
    decoded_msgs = torch.sign(fts) > 0 # b k -> b k
    diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    word_accs = (bit_accs == 1) # b
    norm = torch.norm(fts, dim=-1, keepdim=True) # b d -> b 1
    log_stats = {
        'bit_acc_avg': torch.mean(bit_accs).item(),
        'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
        'norm_avg': torch.mean(norm).item(),
        }

    # attacks = {
    #         'none': lambda x: x,
    #         'crop_01': lambda x: utils_img.center_crop(x, 0.1),
    #         'crop_05': lambda x: utils_img.center_crop(x, 0.5),
    #         'resize_03': lambda x: utils_img.resize(x, 0.3),
    #         'resize_05': lambda x: utils_img.resize(x, 0.5),
    #         'rot_25': lambda x: utils_img.rotate(x, 25),
    #         'rot_90': lambda x: utils_img.rotate(x, 90),
    #         'blur': lambda x: utils_img.gaussian_blur(x, sigma=2.0),
    #         'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
    #         'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
    #     }
    # for name, attack in attacks.items():
    #     fts, (_) = encoder_decoder(imgs, msgs, eval_mode=True, eval_aug=attack)
    #     decoded_msgs = torch.sign(fts) > 0 # b k -> b k
    #     diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
    #     log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

    metric_logger = utils.MetricLogger(delimiter="  ")
    torch.cuda.synchronize()
    for name, loss in log_stats.items():
        metric_logger.update(**{name:loss})
        
metric_logger.synchronize_between_processes()
print("Averaged {} stats:".format('test'), metric_logger)