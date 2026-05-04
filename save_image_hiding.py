import io

import numpy as np
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from critic import *
# from test_model import StegFormer
from model import StegFormer
from thop import profile
import config
from utils.CustomDataset import ImageFolderDataset, ConvertToRGB
from utils.get_pretrain_encoder import get_freeze_encoder
import cv2
from PIL import Image


def compress_image(input_image, quality=85):
    buffer = io.BytesIO()
    input_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


args = config.Args()

transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: convert_to_three_channles(x)),
])

transform_temp = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])


def convert_to_three_channles(image):
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image


# initialization
hiding_encoder, reveal_encoder = get_freeze_encoder()
encoder = StegFormer(256, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8, output_dim=3,
                     pretrain_encoder=hiding_encoder,mode='hide')
decoder = StegFormer(256, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret * 3, pretrain_encoder=reveal_encoder,mode='reveal')

# encoder = StegFormer(256, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8, output_dim=3)
# decoder = StegFormer(256, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret * 3)

# 加载模型
# save_path = args.path + '/checkpoint'
# model_path = f'{save_path}/{args.model_name}.pt'
# model_path = '/home/team01/gxz/projects/Stego/benchmark/pre_trained/test/model_checkpoint_06000.pt'
model_path = '/home/team01/gxz/projects/Stego/benchmark/pre_trained/ot/model_checkpoint_06000.pt'  # normal_model
# model_path = '/home/team01/gxz/projects/Stego/checkpoint/test/model_checkpoint_01000.pt'  # robust_model

state_dicts = torch.load(model_path)
encoder.load_state_dict(state_dicts['encoder'], strict=False)
decoder.load_state_dict(state_dicts['decoder'], strict=False)

encoder.to(args.device)
decoder.to(args.device)

# 计算模型参数量
# with torch.no_grad():
#     test_encoder_input = torch.randn(1, 6, 1024, 1024).to(args.device)
#     test_decoder_input = torch.randn(1, 3, 1024, 1024).to(args.device)
#     encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))
#
#     decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
#     print("thop result:encoder FLOPs=" + str(encoder_mac * 2) + ",encoder params=" + str(encoder_params))
#     print("thop result:decoder FLOPs=" + str(decoder_mac * 2) + ",decoder params=" + str(decoder_params))

# 加载数据集
# data_dir = '/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/data/DIV2K_train_HR'
data_dir = '/home/team01/gxz/projects/Stego/data/Dataset002_Luna16'
test_data_dir = data_dir
test_dataset = ImageFolderDataset(root_dir=test_data_dir, transform=transform, limit=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

w_set = '/home/team01/gxz/projects/Stego/data/val'
water_dataset = ImageFolderDataset(root_dir=w_set, transform=transform, limit=2)
water_loader = DataLoader(water_dataset, batch_size=1, shuffle=False)

i = 0  # 为每一张图编号
# 评价指标
psnr_secret = []
psnr_cover = []
psnr_secret_y = []
psnr_cover_y = []
ssim_secret = []
ssim_cover = []
mse_cover = []
mse_secret = []
rmse_cover = []
rmse_secret = []
mae_cover = []
mae_secret = []

# with clamp
for j in range(1):
    # test 1,000 images
    with torch.no_grad():
        # val
        encoder.eval()
        decoder.eval()

        # 在验证集上测试
        for (cover, secret) in zip(test_loader, water_loader):
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # encode
            msg = torch.cat([cover, secret], 1)
            # encode_img,auxiliary = encoder(msg,None)  # 添加残差连接
            encode_img = encoder(msg)  # 添加残差连接
            encode_img = torch.clamp(encode_img, 0, 1)

            torchvision.utils.save_image(cover, args.path + '/output/ours/robust/cover/' + '%.5d.png' % i)
            torchvision.utils.save_image(secret, args.path + '/output/ours/robust/secret/' + '%.5d.png' % i)

            # robust val
            #  ------------------------------------ rotate  ------------------------------------
            robust_encode_img = torch.rot90(encode_img, k=1, dims=(2, 3))

            # ------------------------------------ compress ------------------------------------
            # image = Image.open('/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/benchmark'
            #                    '/StegFormer/image/clamp/stego/00000.png')
            # encode_img = compress_image(encode_img, quality=50)
            # opearte = transforms.ToTensor()
            # encode_img = opearte(encode_img).unsqueeze(0).to('cuda:0')
            # torchvision.utils.save_image(encode_img, args.path + '/image/clamp/stego/' + '%.5d.png' % i)

            #  ------------------------------------ add noise ------------------------------------
            # std_dev = 0.3
            # noise = torch.randn_like(encode_img) * std_dev
            # robust_encode_img = encode_img + noise

            #  ------------------------------------ cover ------------------------------------
            # encode_img[:,:,150:250,150:250] = 0

            #  ------------------------------------ crop ------------------------------------
            # encode_img = encode_img[:, :, 10:100, 10:100]

            # decode
            # decode_img = decoder(robust_encode_img,auxiliary)
            decode_img = decoder(robust_encode_img)

            # 限制为图像表示
            decode_img = decode_img.clamp(0, 1)
            encode_img = encode_img.clamp(0, 1)

            # 计算各种指标
            # 拷贝进内存以方便计算
            cover = cover.cpu()
            secret = secret.cpu()
            encode_img = encode_img.cpu()
            decode_img = decode_img.cpu()
            robust_encode_img = robust_encode_img.cpu()

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 SSIM
            ssim_encode = calculate_ssim_skimage(cover, encode_img)
            ssim_decode = calculate_ssim_skimage(secret, decode_img)
            ssim_cover.append(ssim_encode)
            ssim_secret.append(ssim_decode)

            # 计算 RMSE
            rmse_cover_temp = calculate_rmse(cover, encode_img)
            rmse_secret_temp = calculate_rmse(secret, decode_img)
            rmse_cover.append(rmse_cover_temp)
            rmse_secret.append(rmse_secret_temp)

            # 计算 MAE
            mae_cover_temp = calculate_mae(cover, encode_img)
            mae_secret_temp = calculate_mae(secret, decode_img)
            mae_cover.append(mae_cover_temp)
            mae_secret.append(mae_secret_temp)

            # 保存图像
            torchvision.utils.save_image(cover, args.path + '/output/ours/robust/cover.png')
            torchvision.utils.save_image(encode_img, args.path + '/output/ours/robust/stego.png')
            torchvision.utils.save_image(secret, args.path + '/output/ours/robust/secret.png')
            torchvision.utils.save_image(decode_img, args.path + '/output/ours/robust/recovery.png')
            torchvision.utils.save_image(robust_encode_img, args.path + '/output/ours/robust/robust.png')
            i += 1  # 下一张图像
            # print("img "+str(i)+" :")
            # print("PSNR_Y_cover:" + str(np.mean(psnry_encode_temp)) + " PSNR_Y_secret:" + str(np.mean(psnry_decode_temp)))
            # print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
            # print("RMSE_cover:" + str(np.mean(rmse_cover_temp)) + " RMSE_secret:" + str(np.mean(rmse_secret_temp)))
            # print("MAE_cover:" + str(np.mean(mae_cover_temp)) + " MAE_secret:" + str(np.mean(mae_secret_temp)))

print("clamp total result:")
print("PSNR_Y_cover:" + str(np.mean(psnr_cover_y)) + " PSNR_Y_secret:" + str(np.mean(psnr_secret_y)))
print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
print("MAE_cover:" + str(np.mean(mae_cover)) + " MAE_secret:" + str(np.mean(mae_secret)))
print("RMSE_cover:" + str(np.mean(rmse_cover)) + " RMSE_secret:" + str(np.mean(rmse_secret)))
