import qrcode
from PIL import Image

import data_augmentation

img = qrcode.make('You are a genius.',box_size=10,version=1,error_correction=qrcode.constants.ERROR_CORRECT_H)
img.save('test.png')
print(img.size)

 
# 定义要生成二维码的内容
data = "Hello, World!ABCDEFGHIJKLMNOPQRST,123987654"
 
# 创建QRCode对象
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4
)

# 将数据添加到QRCode对象中
qr.add_data(data)
qr.make(fit=True)  
   
"""
自动调整二维码的大小以适应数据。当fit参数设置为True时，生成的二维码图像会根据包含的数据自动调整大小，以确保所有数据都能被正确编码到二维码中。
通过设置fit=True参数，可以确保生成的二维码图像适合包含的数据，避免数据被截断或溢出。这样可以保证生成的二维码图像具有最佳的可读性和准确性。
"""

 
# 生成QRCode图像
img = qr.make_image( fill_color="black", back_color="white")
 
# # 添加Logo到二维码
# logo = Image.open("头像.jpg")
# img.paste(logo, (5,5))
 
# 保存生成的二维码图像
img.save("custom_qrcode.png")






# 定义要生成二维码的链接
link = "https://www.marmottan.fr/"
# link = "https://www.marmottan.fr/"
 
# 创建QRCode对象
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
 
# 将链接添加到QRCode对象中
qr.add_data(link)
qr.make(fit=True)
 
# 生成QRCode图像
img = qr.make_image(fill_color="black", back_color="white")
 
# 保存生成的二维码图像
img.save("link_qrcode.png")
