import qrcode
from PIL import Image

import data_augmentation

img = qrcode.make('Hello, world!',box_size=10,version=1,error_correction=qrcode.constants.ERROR_CORRECT_H)
img=img.resize([128,128])
img.save('QR_code/2.png')
print(img.size)
