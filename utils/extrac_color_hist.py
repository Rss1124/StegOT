import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

image = Image.open("/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/data/CELEBAHQ/00000.jpg")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为128x128像素
    transforms.ToTensor(),  # 将图像转换为张量
])
image_tensor = transform(image).squeeze()

r_channel = image_tensor[0]
g_channel = image_tensor[1]
b_channel = image_tensor[2]

r_hist = np.histogram(r_channel.numpy(), bins=256, range=(0,1))
g_hist = np.histogram(g_channel.numpy(), bins=256, range=(0,1))
b_hist = np.histogram(b_channel.numpy(), bins=256, range=(0,1))

plt.figure(figsize=(10,5))
plt.plot(r_hist[1][:-1], r_hist[0], color='red', label='Red')
plt.plot(g_hist[1][:-1], g_hist[0], color='green', label='green')
plt.plot(b_hist[1][:-1], b_hist[0], color='blue', label='blue')
plt.legend()
plt.show()

