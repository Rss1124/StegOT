import torch
import torchvision
from PIL import Image
from torchvision import transforms
import benchmark.Unet_common as common
from benchmark.models.Hinet import *
import config
from critic import *
from test_model import StegFormer as stegformer
from model import StegFormer as ours
from utils.get_pretrain_encoder import get_freeze_encoder
from benchmark.models.HiDDeN import EncoderDecoder, Discriminator
from benchmark.models.WengNet import wengnet


def convert_to_three_channles(image):
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 将图像调整为256x256像素
    transforms.CenterCrop((256, 256)),  # 将图像调整为256x256像素
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: convert_to_three_channles(x))
])


def our(cover, secret):
    rootpath = '/home/team01/gxz/projects/Stego/output'
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

    torchvision.utils.save_image(cover, rootpath + '/ours/cover.png')
    torchvision.utils.save_image(secret, rootpath + '/ours/secrert.png')
    torchvision.utils.save_image(encode_img, rootpath + '/ours/stego.png')
    torchvision.utils.save_image(decode_img, rootpath + '/ours/recover.png')

    print("ours")
    print(calculate_psnr_skimage(cover, encode_img))
    print(calculate_ssim_skimage(cover, encode_img))
    print(calculate_psnr_skimage(secret, decode_img))
    print(calculate_ssim_skimage(secret, decode_img))


def water_sf(cover, secret):
    rootpath = '/home/team01/gxz/projects/Stego/output'
    cover = cover.to(args.device)
    secret = secret.to(args.device)

    # encode
    msg = torch.cat([cover, secret], 1)
    encode_img = encoder(msg)  # 添加残差连接
    encode_img = torch.clamp(encode_img, 0, 1)

    # decode
    decode_img = decoder(encode_img)

    # 限制为图像表示
    decode_img = decode_img.clamp(0, 1)
    encode_img = encode_img.clamp(0, 1)

    cover = cover.cpu()
    secret = secret.cpu()
    encode_img = encode_img.cpu()
    decode_img = decode_img.cpu()

    torchvision.utils.save_image(cover, rootpath + '/sf/cover.png')
    torchvision.utils.save_image(secret, rootpath + '/sf/secrert.png')
    torchvision.utils.save_image(encode_img, rootpath + '/sf/stego.png')
    torchvision.utils.save_image(decode_img, rootpath + '/sf/recover.png')

    print("sf")
    print(calculate_psnr_skimage(cover, encode_img))
    print(calculate_ssim_skimage(cover, encode_img))
    print(calculate_psnr_skimage(secret, decode_img))
    print(calculate_ssim_skimage(secret, decode_img))


def water_hinet(cover, secret):
    rootpath = '/home/team01/gxz/projects/Stego/output'
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

    bacward_img = net(output_rev, rev=True)
    secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
    secret_rev = iwt(secret_rev)

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = steg_img.cpu()
    secret_rev = secret_rev.cpu()

    torchvision.utils.save_image(cover, rootpath + '/hinet/cover.png')
    torchvision.utils.save_image(secret, rootpath + '/hinet/secrert.png')
    torchvision.utils.save_image(steg_img, rootpath + '/hinet/stego.png')
    torchvision.utils.save_image(secret_rev, rootpath + '/hinet/recover.png')

    print("hinet")
    print(calculate_psnr_skimage(cover, steg_img))
    print(calculate_ssim_skimage(cover, steg_img))
    print(calculate_psnr_skimage(secret, secret_rev))
    print(calculate_ssim_skimage(secret, secret_rev))


def water_weng(cover, secret):
    rootpath = '/home/team01/gxz/projects/Stego/output'
    ################## forward ####################
    stego, secret_rev = weng(secret, cover, 'test')

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = stego.cpu()
    secret_rev = secret_rev.cpu()

    torchvision.utils.save_image(cover, rootpath + '/weng/cover.png')
    torchvision.utils.save_image(secret, rootpath + '/weng/secrert.png')
    torchvision.utils.save_image(steg_img, rootpath + '/weng/stego.png')
    torchvision.utils.save_image(secret_rev, rootpath + '/weng/recover.png')

    print("weng")
    print(calculate_psnr_skimage(cover, steg_img))
    print(calculate_ssim_skimage(cover, steg_img))
    print(calculate_psnr_skimage(secret, secret_rev))
    print(calculate_ssim_skimage(secret, secret_rev))


def water_hidden(cover, secret):
    rootpath = '/home/team01/gxz/projects/Stego/output'
    ################## forward ####################
    stego, secret_rev = enc_decoder(cover, secret)

    cover = cover.cpu()
    secret = secret.cpu()
    steg_img = stego.cpu()
    secret_rev = secret_rev.cpu()

    torchvision.utils.save_image(cover, rootpath + '/hidden/cover.png')
    torchvision.utils.save_image(secret, rootpath + '/hidden/secrert.png')
    torchvision.utils.save_image(steg_img, rootpath + '/hidden/stego.png')
    torchvision.utils.save_image(secret_rev, rootpath + '/hidden/recover.png')

    print("hidden")
    print(calculate_psnr_skimage(cover, steg_img))
    print(calculate_ssim_skimage(cover, steg_img))
    print(calculate_psnr_skimage(secret, secret_rev))
    print(calculate_ssim_skimage(secret, secret_rev))


def img_embed(cover, water):
    our(cover, water)
    water_sf(cover, water)
    water_hinet(cover, water)
    water_weng(cover, water)
    water_hidden(cover, water)
    print('done!')


if __name__ == "__main__":
    device = 'cuda:0'  # cuda:0

    # summary(AEmodel, input_size=(1,3,64,64))

    # ----------------------------------------------- cover -----------------------------------------------
    cover = Image.open("/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/data/subset0_/2.png")

    cover = transform(cover).unsqueeze(0).to(device)

    # ----------------------------------------------- watermark -----------------------------------------------
    watermark = Image.open("/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/data/val/ILSVRC2012_val_00000548.JPEG")
    watermark = transform(watermark).unsqueeze(0).to(device)

    """ -------------------------------------------------- steg -------------------------------------------------- """
    args = config.Args()

    # initialization
    encoder = stegformer(256, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8, output_dim=3)
    decoder = stegformer(256, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret * 3)

    # 加载模型
    # save_path = args.path + '/checkpoint'
    # model_path = f'{save_path}/{args.model_name}.pt'
    model_path = '/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/benchmark/StegFormer/checkpoint/test/model_checkpoint_06000.pt'
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)

    encoder.to(args.device).eval()
    decoder.to(args.device).eval()

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
    model_path = '/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/benchmark/StegFormer/checkpoint/ot/model_checkpoint_06000.pt'
    state_dicts = torch.load(model_path)
    our_encoder.load_state_dict(state_dicts['encoder'], strict=False)
    our_decoder.load_state_dict(state_dicts['decoder'], strict=False)

    our_encoder.to(args.device).eval()
    our_decoder.to(args.device).eval()

    """-------------------------------------------------- hidden --------------------------------------------------"""
    enc_decoder = EncoderDecoder().to(device)
    discriminator = Discriminator().to(device)
    enc_decoder.load_state_dict(torch.load('/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/model'
                                           '/benchmark/hidden_checkpoint_2850.pt'))
    enc_decoder.eval()

    """-------------------------------------------------- wengnet --------------------------------------------------"""
    weng = wengnet().to(device)
    weng.load_state_dict(torch.load('/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/model'
                                    '/benchmark/weng_checkpoint_2975.pt'))
    weng.eval()

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

    img_embed(cover=cover, water=watermark)
