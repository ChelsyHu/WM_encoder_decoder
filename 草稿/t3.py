import random 
import qrcode 
import numpy as np
import torch 
import utils_img


from torchvision import transforms
artist_name=['Michelangelo',"Vincent van Gogh","Caravaggio",
   "Rembrandt","Leonardo da Vinci","Johannes Vermeer",
   "Claude Monet","Raphael","Pablo Picasso","Diego Velázquez",
   "Salvador Dalí","Pierre-Auguste Renoir","Peter Paul Rubens",
   "Édouard Manet", "Paul Cézanne", "Edgar Degas",
   "Gustav Klimt", "Edvard Munch", "Henri Matisse","Georges-Pierre Seurat"
   ]

# 设置词组长度
word_count = 8
 
# 随机生成词组
random_words = [random.choice(artist_name) for _ in range(word_count)]
print(random_words)


for i in range(word_count):
    wm= qrcode.make(random_words[i], box_size=10,version=1,error_correction=qrcode.constants.ERROR_CORRECT_H)
    wm =wm.resize([128,128])
    transform = transforms.ToTensor()
    watermark = transform(wm)
    watermark = watermark.unsqueeze(0)  # 1*h*w 

    if i==0:
        watermark_b=watermark
    else:
        watermark_b=np.vstack( [watermark_b, watermark] )  ## b * 1 * h *w 

print(watermark_b.shape) ## 8*1*128*128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
watermark_b=torch.from_numpy(watermark_b).to(device,non_blocking=True)

utils_img.unnormalize_img(watermark_b)

print(watermark_b)

watermark_b.save('QR_code/all.png')



