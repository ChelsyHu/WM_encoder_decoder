import os

def count_images(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            count += 1
    return count

# 使用函数
folder_path = 'datasets/coco_dataset/train2014'  # 请替换为你的文件夹路径
num_images = count_images(folder_path)
print(f'文件夹中图片的个数是：{num_images}')

folder_path = 'datasets/coco_dataset/test2014'  # 请替换为你的文件夹路径
num_images = count_images(folder_path)
print(f'文件夹中图片的个数是：{num_images}')