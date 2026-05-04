import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import benchmark.Unet_common as common
from benchmark.models.WengNet import wengnet
import config
from critic import *
from test_model import StegFormer as stegformer
from model import StegFormer as ours
from benchmark.models.HiDDeN import EncoderDecoder, Discriminator
from utils.get_pretrain_encoder import get_freeze_encoder
from benchmark.models.Hinet import *
import benchmark.config as c
from utils.CustomDataset import ConvertToRGB, ImageFolderDataset
import torchvision.models as models


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为256x256像素
    ConvertToRGB(),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: convert_to_three_channles(x))
    # ConvertToRGB(),
    # transforms.CenterCrop((256, 256)),
    # transforms.ToTensor(),  # 将图像转换为张量
    # transforms.Lambda(lambda x: convert_to_three_channles(x)),
])


def convert_to_three_channles(image):
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def our(cover, secret, index):
    cover = cover.to(args.device)
    secret = secret.to(args.device)

    # encode
    msg = torch.cat([cover, secret], 1)
    encode_img = our_encoder(msg)  # 添加残差连接
    encode_img = torch.clamp(encode_img, 0, 1)

    # decode
    decode_img = our_decoder(encode_img)

    # 限制为图像表示
    decode_img = decode_img.clamp(0, 1)
    encode_img = encode_img.clamp(0, 1)

    cover = cover.cpu()
    secret = secret.cpu()
    encode_img = encode_img.cpu()
    decode_img = decode_img.cpu()

    our_total_cc_psnr.append(calculate_psnr_skimage(cover, encode_img))
    our_total_cc_ssim.append(calculate_ssim_skimage(cover, encode_img))
    our_total_cc_mae.append(calculate_mae(cover, encode_img))
    our_total_cc_rmse.append(calculate_rmse(cover, encode_img))
    our_total_cc_perceptual.append(calculate_perceptual_loss(cover,encode_img))

    our_total_ss_psnr.append(calculate_psnr_skimage(secret, decode_img))
    our_total_ss_ssim.append(calculate_ssim_skimage(secret, decode_img))
    our_total_ss_mae.append(calculate_mae(secret, decode_img))
    our_total_ss_rmse.append(calculate_rmse(secret, decode_img))
    our_total_ss_perceptual.append(calculate_perceptual_loss(secret,decode_img))

    ori_img = np.transpose(cover.detach().squeeze().cpu().numpy(), (1, 2, 0))
    ori_img = np.clip(ori_img, 0, 255)

    ori_water = np.transpose(secret.detach().squeeze().cpu().numpy(), (1, 2, 0))
    ori_water = np.clip(ori_water, 0, 255)

    npc_img_encode = np.transpose(encode_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    npc_img_encode = np.clip(npc_img_encode, 0, 255)

    npc_water_encode = np.transpose(decode_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    npc_water_encode = np.clip(npc_water_encode, 0, 255)

    # if index < 0:
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(ori_img)
    #     plt.title('ori_img', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(ori_water)
    #     plt.title('ori_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(npc_img_encode)
    #     plt.title('image_with_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(npc_water_encode)
    #     plt.title('water_from_embed_img', color='white')
    #     plt.axis('off')
    #
    #     plt.savefig(out_dir + 'water_extract_' + str('{:05d}'.format(index)) + '.png', bbox_inches='tight',
    #                 pad_inches=0,
    #                 transparent=True)


def water_sf(cover, secret, index):
    cover = cover.to(args.device)
    secret = secret.to(args.device)

    # encode
    msg = torch.cat([cover, secret], 1)
    encode_img = encoder(msg)  # 添加残差连接
    encode_img = torch.clamp(encode_img, 0, 1)

    # # add noise
    # std_dev = 0.05
    # noise = torch.randn_like(encode_img) * std_dev
    # encode_img = encode_img + noise
    # encode_img = torch.clamp(encode_img, 0, 1)

    # decode
    decode_img = decoder(encode_img)

    # 限制为图像表示
    decode_img = decode_img.clamp(0, 1)
    encode_img = encode_img.clamp(0, 1)

    cover = cover.cpu()
    secret = secret.cpu()
    encode_img = encode_img.cpu()
    decode_img = decode_img.cpu()

    sf_total_cc_psnr.append(calculate_psnr_skimage(cover, encode_img))
    sf_total_cc_ssim.append(calculate_ssim_skimage(cover, encode_img))
    sf_total_cc_mae.append(calculate_mae(cover, encode_img))
    sf_total_cc_rmse.append(calculate_rmse(cover, encode_img))
    sf_total_cc_perceptual.append(calculate_perceptual_loss(cover,encode_img))

    sf_total_ss_psnr.append(calculate_psnr_skimage(secret, decode_img))
    sf_total_ss_ssim.append(calculate_ssim_skimage(secret, decode_img))
    sf_total_ss_mae.append(calculate_mae(secret, decode_img))
    sf_total_ss_rmse.append(calculate_rmse(secret, decode_img))
    sf_total_ss_perceptual.append(calculate_perceptual_loss(secret,decode_img))

    ori_img = np.transpose(cover.detach().squeeze().cpu().numpy(), (1, 2, 0))
    ori_img = np.clip(ori_img, 0, 255)

    ori_water = np.transpose(secret.detach().squeeze().cpu().numpy(), (1, 2, 0))
    ori_water = np.clip(ori_water, 0, 255)

    npc_img_encode = np.transpose(encode_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    npc_img_encode = np.clip(npc_img_encode, 0, 255)

    npc_water_encode = np.transpose(decode_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    npc_water_encode = np.clip(npc_water_encode, 0, 255)

    # if index < 0:
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(ori_img)
    #     plt.title('ori_img', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(ori_water)
    #     plt.title('ori_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(npc_img_encode)
    #     plt.title('image_with_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(npc_water_encode)
    #     plt.title('water_from_embed_img', color='white')
    #     plt.axis('off')
    #
    #     plt.savefig(out_dir + 'water_extract_' + str('{:05d}'.format(index)) + '.png', bbox_inches='tight',
    #                 pad_inches=0,
    #                 transparent=True)


def water_hinet(cover, secret, index):
    cover_input = dwt(cover)
    secret_input = dwt(secret)
    input_img = torch.cat((cover_input, secret_input), 1)

    #################
    #    forward:   #
    #################
    output = net(input_img)
    output_steg = output.narrow(1, 0, 4 * c.channels_in)
    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
    steg_img = iwt(output_steg)
    backward_z = gauss_noise(output_z.shape)

    #################
    #   backward:   #
    #################
    output_rev = torch.cat((output_steg, backward_z), 1)

    # add noise
    # std_dev = 0.05
    # noise = torch.randn_like(output_rev) * std_dev
    # output_rev = output_rev + noise
    # output_rev = torch.clamp(output_rev, 0, 1)

    bacward_img = net(output_rev, rev=True)
    secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
    secret_rev = iwt(secret_rev)

    torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
    torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
    torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
    torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)

    ori_img = np.transpose(cover.detach().squeeze().cpu().numpy(), (1, 2, 0))
    # ori_img = np.clip(ori_img, 0, 255)

    ori_water = np.transpose(secret.detach().squeeze().cpu().numpy(), (1, 2, 0))
    # ori_water = np.clip(ori_water, 0, 255)

    image_with_water = np.transpose(steg_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    # image_with_water = np.clip(image_with_water, 0, 255)

    water_from_embed_img = np.transpose(secret_rev.detach().squeeze().cpu().numpy(), (1, 2, 0))
    # water_from_embed_img = np.clip(water_from_embed_img, 0, 255)

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = steg_img.cpu()
    secret_rev = secret_rev.cpu()

    hinet_total_cc_psnr.append(calculate_psnr_skimage(cover, steg_img))
    hinet_total_cc_ssim.append(calculate_ssim_skimage(cover, steg_img))
    hinet_total_cc_mae.append(calculate_mae(cover, steg_img))
    hinet_total_cc_rmse.append(calculate_rmse(cover, steg_img))
    hinet_total_cc_perceptual.append(calculate_perceptual_loss(cover,steg_img))
    hinet_total_ss_psnr.append(calculate_psnr_skimage(secret, secret_rev))
    hinet_total_ss_ssim.append(calculate_ssim_skimage(secret, secret_rev))
    hinet_total_ss_mae.append(calculate_mae(secret, secret_rev))
    hinet_total_ss_rmse.append(calculate_rmse(secret, secret_rev))
    hinet_total_ss_perceptual.append(calculate_perceptual_loss(secret,secret_rev))

    # if index < 0:
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(ori_img)
    #     plt.title('ori_img', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(ori_water)
    #     plt.title('ori_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(image_with_water)
    #     plt.title('image_with_water', color='white')
    #     plt.axis('off')
    #
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(water_from_embed_img)
    #     plt.title('water_from_embed_img', color='white')
    #     plt.axis('off')
    #
    #     plt.savefig(rootpath + 'water_extract_' + str('{:05d}'.format(i)) + '.png', bbox_inches='tight',
    #                 pad_inches=0,
    #                 transparent=True)

def water_weng(cover, secret):
    ################## forward ####################
    stego, secret_rev = weng(secret, cover, 'test')

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = stego.cpu()
    secret_rev = secret_rev.cpu()

    weng_total_cc_psnr.append(calculate_psnr_skimage(cover, steg_img))
    weng_total_cc_ssim.append(calculate_ssim_skimage(cover, steg_img))
    weng_total_cc_mae.append(calculate_mae(cover, steg_img))
    weng_total_cc_rmse.append(calculate_rmse(cover, steg_img))
    weng_total_cc_perceptual.append(calculate_perceptual_loss(cover,steg_img))
    weng_total_ss_psnr.append(calculate_psnr_skimage(secret, secret_rev))
    weng_total_ss_ssim.append(calculate_ssim_skimage(secret, secret_rev))
    weng_total_ss_mae.append(calculate_mae(secret, secret_rev))
    weng_total_ss_rmse.append(calculate_rmse(secret, secret_rev))
    weng_total_ss_perceptual.append(calculate_perceptual_loss(secret,secret_rev))


def water_hidden(cover, secret):
    ################## forward ####################
    stego, secret_rev = enc_decoder(cover, secret)

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = stego.cpu()
    secret_rev = secret_rev.cpu()

    hidden_total_cc_psnr.append(calculate_psnr_skimage(cover, steg_img))
    hidden_total_cc_ssim.append(calculate_ssim_skimage(cover, steg_img))
    hidden_total_cc_mae.append(calculate_mae(cover, steg_img))
    hidden_total_cc_rmse.append(calculate_rmse(cover, steg_img))
    hidden_total_cc_perceptual.append(calculate_perceptual_loss(cover,steg_img))
    hidden_total_ss_psnr.append(calculate_psnr_skimage(secret, secret_rev))
    hidden_total_ss_ssim.append(calculate_ssim_skimage(secret, secret_rev))
    hidden_total_ss_mae.append(calculate_mae(secret, secret_rev))
    hidden_total_ss_rmse.append(calculate_rmse(secret, secret_rev))
    hidden_total_ss_perceptual.append(calculate_perceptual_loss(secret,secret_rev))


# data_names = ['COCO2017', 'DIV2K_train_HR', 'ILSVRC2012_img_val']
data_names = ['COVID19', 'IQ-OTHNCCD']
device = 'cuda:0'  # cuda:0

vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

for data_name in data_names:
    """ --------------------------------------------------- data --------------------------------------------------- """
    data_dir = '/home/team01/gxz/projects/Stego/data/' + data_name
    test_data_dir = data_dir
    test_dataset = ImageFolderDataset(root_dir=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    t = tqdm(test_loader, desc=f'[overall progress]')

    # wm_loaders
    w_set = '/home/team01/gxz/projects/Stego/data/val'
    water_dataset = ImageFolderDataset(root_dir=w_set, transform=transform)
    water_loader = DataLoader(water_dataset, batch_size=1, shuffle=True)
    temp_water_loader = water_loader

    plt.figure(figsize=(10, 8))  # 图像尺寸

    """-------------------------------------------------- hidden --------------------------------------------------"""
    enc_decoder = EncoderDecoder().to(device)
    discriminator = Discriminator().to(device)
    enc_decoder.load_state_dict(torch.load('/home/team01/gxz/projects/Stego/benchmark/pre_trained/hidden_checkpoint_2850.pt'))
    enc_decoder.eval()
    hidden_total_cc_psnr = []
    hidden_total_cc_ssim = []
    hidden_total_cc_mae = []
    hidden_total_cc_rmse = []
    hidden_total_cc_perceptual = []
    hidden_total_ss_psnr = []
    hidden_total_ss_ssim = []
    hidden_total_ss_mae = []
    hidden_total_ss_rmse = []
    hidden_total_ss_perceptual = []


    """-------------------------------------------------- wengnet --------------------------------------------------"""
    weng = wengnet().to(device)
    weng.load_state_dict(torch.load('/home/team01/gxz/projects/Stego/benchmark/pre_trained/weng_checkpoint_2975.pt'))
    weng.eval()
    weng_total_cc_psnr = []
    weng_total_cc_ssim = []
    weng_total_cc_mae = []
    weng_total_cc_rmse = []
    weng_total_cc_perceptual = []
    weng_total_ss_psnr = []
    weng_total_ss_ssim = []
    weng_total_ss_mae = []
    weng_total_ss_rmse = []
    weng_total_ss_perceptual = []

    """--------------------------------------------------- hinet ---------------------------------------------------"""
    net = Model()
    net.cuda()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    load(c.MODEL_PATH + c.suffix)
    net.eval()
    dwt = common.DWT()
    iwt = common.IWT()

    hinet_total_cc_psnr = []
    hinet_total_cc_ssim = []
    hinet_total_cc_mae = []
    hinet_total_cc_rmse = []
    hinet_total_cc_perceptual = []
    hinet_total_ss_psnr = []
    hinet_total_ss_ssim = []
    hinet_total_ss_mae = []
    hinet_total_ss_rmse = []
    hinet_total_ss_perceptual = []

    """ -------------------------------------------------- steg -------------------------------------------------- """
    args = config.Args()

    # initialization
    encoder = stegformer(256, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8, output_dim=3)
    decoder = stegformer(256, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret * 3)

    # 加载模型
    # save_path = args.path + '/checkpoint'
    # model_path = f'{save_path}/{args.model_name}.pt'
    model_path = '/home/team01/gxz/projects/Stego/benchmark/pre_trained/test/model_checkpoint_06000.pt'
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)

    encoder.to(args.device).eval()
    decoder.to(args.device).eval()

    sf_total_cc_psnr = []
    sf_total_cc_ssim = []
    sf_total_cc_mae = []
    sf_total_cc_rmse = []
    sf_total_cc_perceptual = []
    sf_total_ss_psnr = []
    sf_total_ss_ssim = []
    sf_total_ss_mae = []
    sf_total_ss_rmse = []
    sf_total_ss_perceptual = []

    """ -------------------------------------------------- our -------------------------------------------------- """
    args = config.Args()
    hiding_encoder, reveal_encoder = get_freeze_encoder()
    # initialization
    our_encoder = ours(256, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8, output_dim=3,
                       pretrain_encoder=hiding_encoder)
    our_decoder = ours(256, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret * 3, pretrain_encoder=reveal_encoder)

    # 加载模型
    # save_path = args.path + '/checkpoint'
    # model_path = f'{save_path}/{args.model_name}.pt'
    model_path = '/home/team01/gxz/projects/Stego/benchmark/pre_trained/ot/model_checkpoint_06000.pt'
    state_dicts = torch.load(model_path)
    our_encoder.load_state_dict(state_dicts['encoder'], strict=False)
    our_decoder.load_state_dict(state_dicts['decoder'], strict=False)

    our_encoder.to(args.device).eval()
    our_decoder.to(args.device).eval()

    our_total_cc_psnr = []
    our_total_cc_ssim = []
    our_total_cc_mae = []
    our_total_cc_rmse = []
    our_total_cc_perceptual = []
    our_total_ss_psnr = []
    our_total_ss_ssim = []
    our_total_ss_mae = []
    our_total_ss_rmse = []
    our_total_ss_perceptual = []

    """ ------------------------------------------------- benchmark ------------------------------------------------- """

    with torch.no_grad():
        i = 0
        for data in t:
            if (i + 1) % len(water_loader) == 0:
                temp_water_loader = water_loader

            data = data.to(device)

            cover = data
            secret = next(iter(temp_water_loader)).to(device)

            # cover = data[data.shape[0] // 2:, :, :, :]
            # secret = data[:data.shape[0] // 2, :, :, :]
            temp_water_loader = iter(temp_water_loader)

            water_hidden(cover, secret)
            water_weng(cover, secret)
            water_sf(cover, secret, i)
            water_hinet(cover, secret, i)
            our(cover, secret, i)

            i = i + 1


    hinet_average_cc_psnr = sum(hinet_total_cc_psnr) / len(hinet_total_cc_psnr)
    hinet_average_cc_ssim = sum(hinet_total_cc_ssim) / len(hinet_total_cc_ssim)
    hinet_average_cc_mae = sum(hinet_total_cc_mae) / len(hinet_total_cc_mae)
    hinet_average_cc_rmse = sum(hinet_total_cc_rmse) / len(hinet_total_cc_rmse)
    hinet_average_cc_perceptual = sum(hinet_total_cc_perceptual) / len(hinet_total_cc_perceptual)
    hinet_average_ss_psnr = sum(hinet_total_ss_psnr) / len(hinet_total_ss_psnr)
    hinet_average_ss_ssim = sum(hinet_total_ss_ssim) / len(hinet_total_ss_ssim)
    hinet_average_ss_mae = sum(hinet_total_ss_mae) / len(hinet_total_ss_mae)
    hinet_average_ss_rmse = sum(hinet_total_ss_rmse) / len(hinet_total_ss_rmse)
    hinet_average_ss_perceptual = sum(hinet_total_ss_perceptual) / len(hinet_total_ss_perceptual)

    sf_average_cc_psnr = sum(sf_total_cc_psnr) / len(sf_total_cc_psnr)
    sf_average_cc_ssim = sum(sf_total_cc_ssim) / len(sf_total_cc_ssim)
    sf_average_cc_mae = sum(sf_total_cc_mae) / len(sf_total_cc_mae)
    sf_average_cc_rmse = sum(sf_total_cc_rmse) / len(sf_total_cc_rmse)
    sf_average_cc_perceptual = sum(sf_total_cc_perceptual) / len(sf_total_cc_perceptual)
    sf_average_ss_psnr = sum(sf_total_ss_psnr) / len(sf_total_ss_psnr)
    sf_average_ss_ssim = sum(sf_total_ss_ssim) / len(sf_total_ss_ssim)
    sf_average_ss_mae = sum(sf_total_ss_mae) / len(sf_total_ss_mae)
    sf_average_ss_rmse = sum(sf_total_ss_rmse) / len(sf_total_ss_rmse)
    sf_average_ss_perceptual = sum(sf_total_ss_perceptual) / len(sf_total_ss_perceptual)

    our_average_cc_psnr = sum(our_total_cc_psnr) / len(our_total_cc_psnr)
    our_average_cc_ssim = sum(our_total_cc_ssim) / len(our_total_cc_ssim)
    our_average_cc_mae = sum(our_total_cc_mae) / len(our_total_cc_mae)
    our_average_cc_rmse = sum(our_total_cc_rmse) / len(our_total_cc_rmse)
    our_average_cc_perceptual = sum(our_total_cc_perceptual) / len(our_total_cc_perceptual)
    our_average_ss_psnr = sum(our_total_ss_psnr) / len(our_total_ss_psnr)
    our_average_ss_ssim = sum(our_total_ss_ssim) / len(our_total_ss_ssim)
    our_average_ss_mae = sum(our_total_ss_mae) / len(our_total_ss_mae)
    our_average_ss_rmse = sum(our_total_ss_rmse) / len(our_total_ss_rmse)
    our_average_ss_perceptual = sum(our_total_ss_perceptual) / len(our_total_ss_perceptual)

    weng_average_cc_psnr = sum(weng_total_cc_psnr) / len(weng_total_cc_psnr)
    weng_average_cc_ssim = sum(weng_total_cc_ssim) / len(weng_total_cc_ssim)
    weng_average_cc_mae = sum(weng_total_cc_mae) / len(weng_total_cc_mae)
    weng_average_cc_rmse = sum(weng_total_cc_rmse) / len(weng_total_cc_rmse)
    weng_average_cc_perceptual = sum(weng_total_cc_perceptual) / len(weng_total_cc_perceptual)
    weng_average_ss_psnr = sum(weng_total_ss_psnr) / len(weng_total_ss_psnr)
    weng_average_ss_ssim = sum(weng_total_ss_ssim) / len(weng_total_ss_ssim)
    weng_average_ss_mae = sum(weng_total_ss_mae) / len(weng_total_ss_mae)
    weng_average_ss_rmse = sum(weng_total_ss_rmse) / len(weng_total_ss_rmse)
    weng_average_ss_perceptual = sum(weng_total_ss_perceptual) / len(weng_total_ss_perceptual)

    hidden_average_cc_psnr = sum(hidden_total_cc_psnr) / len(hidden_total_cc_psnr)
    hidden_average_cc_ssim = sum(hidden_total_cc_ssim) / len(hidden_total_cc_ssim)
    hidden_average_cc_mae = sum(hidden_total_cc_mae) / len(hidden_total_cc_mae)
    hidden_average_cc_rmse = sum(hidden_total_cc_rmse) / len(hidden_total_cc_rmse)
    hidden_average_cc_perceptual = sum(hidden_total_cc_perceptual) / len(hidden_total_cc_perceptual)
    hidden_average_ss_psnr = sum(hidden_total_ss_psnr) / len(hidden_total_ss_psnr)
    hidden_average_ss_ssim = sum(hidden_total_ss_ssim) / len(hidden_total_ss_ssim)
    hidden_average_ss_mae = sum(hidden_total_ss_mae) / len(hidden_total_ss_mae)
    hidden_average_ss_rmse = sum(hidden_total_ss_rmse) / len(hidden_total_ss_rmse)
    hidden_average_ss_perceptual = sum(hidden_total_ss_perceptual) / len(hidden_total_ss_perceptual)

    print(data_name)

    print("hidden:")
    print("cover/stego pair")
    print(hidden_average_cc_psnr)
    print(hidden_average_cc_ssim)
    print(hidden_average_cc_mae)
    print(hidden_average_cc_rmse)
    print(hidden_average_cc_perceptual)  #LIPIS
    print("secret/rec pair")
    print(hidden_average_ss_psnr)
    print(hidden_average_ss_ssim)
    print(hidden_average_ss_mae)
    print(hidden_average_ss_rmse)
    print(hidden_average_ss_perceptual)

    print("weng:")
    print("cover/stego pair")
    print(weng_average_cc_psnr)
    print(weng_average_cc_ssim)
    print(weng_average_cc_mae)
    print(weng_average_cc_rmse)
    print(weng_average_cc_perceptual)
    print("secret/rec pair")
    print(weng_average_ss_psnr)
    print(weng_average_ss_ssim)
    print(weng_average_ss_mae)
    print(weng_average_ss_rmse)
    print(weng_average_ss_perceptual)

    print("hinet:")
    print("cover/stego pair")
    print(hinet_average_cc_psnr)
    print(hinet_average_cc_ssim)
    print(hinet_average_cc_mae)
    print(hinet_average_cc_rmse)
    print(hinet_average_cc_perceptual)
    print("secret/rec pair")
    print(hinet_average_ss_psnr)
    print(hinet_average_ss_ssim)
    print(hinet_average_ss_mae)
    print(hinet_average_ss_rmse)
    print(hinet_average_ss_perceptual)

    print("steg")
    print("cover/stego pair")
    print(sf_average_cc_psnr)
    print(sf_average_cc_ssim)
    print(sf_average_cc_mae)
    print(sf_average_cc_rmse)
    print(sf_average_cc_perceptual)
    print("secret/rec pair")
    print(sf_average_ss_psnr)
    print(sf_average_ss_ssim)
    print(sf_average_ss_mae)
    print(sf_average_ss_rmse)
    print(sf_average_ss_perceptual)

    print("our:")
    print("cover/stego pair")
    print(our_average_cc_psnr)
    print(our_average_cc_ssim)
    print(our_average_cc_mae)
    print(our_average_cc_rmse)
    print(our_average_cc_perceptual)
    print("secret/rec pair")
    print(our_average_ss_psnr)
    print(our_average_ss_ssim)
    print(our_average_ss_mae)
    print(our_average_ss_rmse)
    print(our_average_ss_perceptual)
