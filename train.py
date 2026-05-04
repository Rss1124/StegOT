import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from critic import *
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from model import StegFormer
import os
import timm.scheduler
import config
from utils.CustomDataset import ImageFolderDataset, ConvertToRGB
from utils.get_pretrain_encoder import get_freeze_encoder

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions()

args = config.Args()

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# loss function
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Restrict_Loss(nn.Module):
    """Restrict loss using L2 loss function"""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, X):
        count1 = torch.sum(X > 1)
        count0 = torch.sum(X < 0)
        if count1 == 0:
            count1 = 1
        if count0 == 0:
            count0 = 1
        one = torch.ones_like(X)
        zero = torch.zeros_like(X)
        X_one = torch.where(X <= 1, 1, X)  # 对超过 1 的值施加惩罚
        X_zero = torch.where(X >= 0, 0, X)  # 对小于 0 的值施加惩罚
        diff_one = X_one - one
        diff_zero = zero - X_zero
        loss = torch.sum(0.5 * (diff_one ** 2)) / count1 + torch.sum(0.5 * (diff_zero ** 2)) / count0
        return loss


transform_train = transforms.Compose([
    ConvertToRGB(),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 90)),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: convert_to_three_channles(x)),
])

transform_val = transforms.Compose([
    ConvertToRGB(),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: convert_to_three_channles(x)),
])


def convert_to_three_channles(image):
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image


hiding_encoder, reveal_encoder = get_freeze_encoder()

# 新建文件夹
model_version_name = 'test'
save_path = args.path + '/checkpoint/' + model_version_name  # 新建一个以模型版本名为名字的文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)

# tensorboard
writer = SummaryWriter('/home/team01/gxz/projects/Stego/watermark/logs/stegformer')

# StegFormer initiate
encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8,
                     output_dim=3, drop_key=False, patch_size=2, window_size=8, output_act=args.output_act,mode='hide',
                     depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2], pretrain_encoder=hiding_encoder)
decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=8, output_dim=3, drop_key=False,
                     patch_size=2, window_size=8, output_act=args.output_act,pretrain_encoder=reveal_encoder,mode='reveal',
                     depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])

# encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8,
#                      output_dim=3, drop_key=False, patch_size=2, window_size=8, output_act=args.output_act,
#                      depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
# decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=8, output_dim=3, drop_key=False,
#                      patch_size=2, window_size=8, output_act=args.output_act,
#                      depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])


encoder.cuda()
decoder.cuda()

# loading model
if args.train_next != 0:
    model_path = save_path + '/model_checkpoint_%.5i' % args.train_next + '.pt'
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)

# optimer and the learning rate scheduler
optim = torch.optim.AdamW([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr)
if args.train_next != 0:
    optim.load_state_dict(state_dicts['opt'])
scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
                                             t_initial=args.epochs,
                                             lr_min=0,
                                             warmup_t=args.warm_up_epoch,
                                             warmup_lr_init=args.warm_up_lr_init)

# loss function
conceal_loss_function = L1_Charbonnier_loss().to(args.device)
reveal_loss_function = L1_Charbonnier_loss().to(args.device)
restrict_loss_function = Restrict_Loss().to(args.device)
vgg = models.vgg19(pretrained=True).features.to("cuda:0").eval()

# train_loaders
data_name = 'DIV2K_train_HR'
data_dir = '/home/team01/gxz/projects/Stego/data/' + data_name
test_data_dir = data_dir

# val_loaders
w_set = '/home/team01/gxz/projects/Stego/data/val'

# train
for i_epoch in range(args.epochs):
    sum_loss = []
    scheduler.step(i_epoch + args.train_next)
    test_dataset = ImageFolderDataset(root_dir=test_data_dir, transform=transform_train, limit=2000)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    t = tqdm(test_loader, desc=f'[signal epoch progress]')
    for data in t:
        cover = data[data.shape[0] // 2:].to(args.device)
        secret = data[:data.shape[0] // 2].to(args.device)

        # encode
        msg = torch.cat([cover, secret], 1)
        encode_img, auxiliary = encoder(msg)

        # normalizing
        if args.norm_train == 'clamp':
            encode_img_c = torch.clamp(encode_img, 0, 1)
        else:
            encode_img_c = encode_img

        #  ------------------------------------ add noise ------------------------------------
        std_dev = random.random() * 0.1
        noise = torch.randn_like(encode_img_c) * std_dev
        noisy_encode_img = encode_img_c + noise

        # decode
        decode_img = decoder(noisy_encode_img, auxiliary)

        # loss
        conceal_loss = conceal_loss_function(cover.cuda(), encode_img.cuda())
        reveal_loss = reveal_loss_function(secret.cuda(), decode_img.cuda())
        # input_C_features = vgg(cover)
        # output_C_features = vgg(encode_img)
        # input_S_features = vgg(secret)
        # output_S_features = vgg(decode_img)
        # l_perceptual_stego = torch.mean(torch.abs(input_C_features - output_C_features))
        # l_perceptual_rec = torch.mean(torch.abs(input_S_features - output_S_features))

        total_loss = None
        if args.norm_train:
            restrict_loss = restrict_loss_function(encode_img.cuda())
            # total_loss = conceal_loss + reveal_loss + restrict_loss + l_perceptual_stego + l_perceptual_rec + emd_img_img + emd_water_water
            total_loss = conceal_loss + reveal_loss + restrict_loss
        else:
            # total_loss = conceal_loss + reveal_loss + l_perceptual_stego + l_perceptual_rec + emd_img_img + emd_water_water
            total_loss = conceal_loss + reveal_loss
        sum_loss.append(total_loss.item())

        # backward
        total_loss.backward()
        optim.step()
        optim.zero_grad()

    # valid
    if i_epoch % args.val_freq == 0:
        print("validation begin:")
        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            # psnr and ssim
            psnr_secret = []
            psnr_cover = []
            ssim_secret = []
            ssim_cover = []
            val_dataset = ImageFolderDataset(root_dir=w_set, transform=transform_val, limit=100)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
            # 在验证集上测试
            for index, val_data in enumerate(val_loader):
                cover = val_data[val_data.shape[0] // 2:].to(args.device)
                secret = val_data[:val_data.shape[0] // 2].to(args.device)

                # encode
                msg = torch.cat([cover, secret], 1)
                encode_img,auxiliary = encoder(msg, None)

                if args.norm_train:
                    encode_img = torch.clamp(encode_img, 0, 1)

                #  ------------------------------------ add noise ------------------------------------
                std_dev = random.random() * 0.1
                noise = torch.randn_like(encode_img) * std_dev
                noisy_encode_img = encode_img + noise

                # decode
                decode_img = decoder(noisy_encode_img,auxiliary)

                encode_img = torch.clamp(encode_img, 0, 1)
                decode_img = torch.clamp(decode_img, 0, 1)

                cover = cover.cpu()
                secret = secret.cpu()
                encode_img = encode_img.cpu()
                decode_img = decode_img.cpu()

                psnr_encode_temp = calculate_psnr(cover, encode_img)
                psnr_decode_temp = calculate_psnr(secret, decode_img)
                psnr_cover.append(psnr_encode_temp)
                psnr_secret.append(psnr_decode_temp)

                ssim_encode_temp = calculate_ssim_skimage(cover, encode_img)
                ssim_decode_temp = calculate_ssim_skimage(secret, decode_img)
                ssim_cover.append(ssim_encode_temp)
                ssim_secret.append(ssim_decode_temp)

            writer.add_scalar("ccpsnr", np.mean(psnr_cover), i_epoch + args.train_next)
            writer.add_scalar("sspsnr", np.mean(psnr_secret), i_epoch + args.train_next)
            writer.add_scalar("ccSSIM", np.mean(ssim_cover), i_epoch + args.train_next)
            writer.add_scalar("ssSSIM", np.mean(ssim_secret), i_epoch + args.train_next)
            print("PSNR_cover:" + str(np.mean(psnr_cover)) + " PSNR_secret:" + str(np.mean(psnr_secret)))
            print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))

    print("epoch:" + str(i_epoch + args.train_next) + ":" + str(np.mean(sum_loss)))
    if i_epoch % 2 == 0:
        writer.add_scalar("loss", np.mean(sum_loss), i_epoch + args.train_next)

    # 保存当前模型以及优化器参数
    if (i_epoch % args.save_freq) == 0:
        torch.save({'opt': optim.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict()},
                   save_path + '/model_checkpoint_%.5i' % (i_epoch + args.train_next) + '.pt')

torch.save({'opt': optim.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()}, f'{save_path}/{model_version_name}.pt')
