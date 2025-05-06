import os
import pytorch_fid.fid_score as fid_score
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2

# 设定路径
path_real_images = 'ckpts/1/test/original_imgs'
path_generated_images = 'ckpts/1/test/watermarked_imgs'


# 列出两个文件夹中的所有图片
real_images = {os.path.basename(f): f for f in os.listdir(path_real_images) if os.path.isfile(os.path.join(path_real_images, f))}
generated_images = {os.path.basename(f): os.path.join(path_generated_images, f) for f in os.listdir(path_generated_images) if os.path.isfile(os.path.join(path_generated_images, f))}


fid_value = fid_score.calculate_fid_given_paths([real_path, generated_path], device='cuda', dims=2048)
fid_values.append(fid_value)


# 找出匹配的图片对
matched_images = {k: real_images[k] for k in real_images if k in generated_images}


# 初始化FID和SSIM的累加值
fid_values = []
ssim_values = []

# 对匹配的图片对进行计算
for name, real_path in matched_images.items():
    generated_path = generated_images[name]
    
    # 计算 FID (这里需要使用pytorch_fid的函数，可能需要稍微调整)
    try:
        fid_value = fid_score.calculate_fid_given_paths([real_path, generated_path], device='cuda', dims=2048)
        fid_values.append(fid_value)
    except RuntimeError as e:
        print(f"计算FID时出错: {e}")

    # 计算 SSIM
    real_img = cv2.imread(real_path)
    generated_img = cv2.imread(generated_path)
    ssim_value = ssim(real_img, generated_img, multichannel=True)
    ssim_values.append(ssim_value)

# 计算平均值
average_fid = np.mean(fid_values) if fid_values else float('nan')
average_ssim = np.mean(ssim_values) if ssim_values else float('nan')

print(f'平均FID: {average_fid}')
print(f'平均SSIM: {average_ssim}')

