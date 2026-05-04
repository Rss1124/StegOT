import torch
import torch.nn as nn
from test_model import StegFormer
import config


class FreezeEncoder(nn.Sequential):
    def __init__(self, encoder):
        super(FreezeEncoder, self).__init__()
        m = []
        m.append(encoder.embedding)
        m.append(encoder.encoderlayer_0)
        m.append(encoder.downsampler_0)
        m.append(encoder.encoderlayer_1)
        m.append(encoder.downsampler_1)
        m.append(encoder.encoderlayer_2)
        m.append(encoder.downsampler_2)
        m.append(encoder.encoderlayer_3)
        m.append(encoder.downsampler_3)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


def get_freeze_encoder():
    args = config.Args()
    model_path = ('/home/team01/gxz/projects/pytorch_diffusion_model_celebahq-master/benchmark/StegFormer/checkpoint'
                  '/StegFormer-S_baseline.pt')
    encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret + 1) * 3, cnn_emb_dim=8,
                         output_dim=3, drop_key=False, patch_size=2, window_size=8, output_act=args.output_act,
                         depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=8, output_dim=3, drop_key=False,
                         patch_size=2, window_size=8, output_act=args.output_act,
                         depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    encoder.cuda()
    decoder.cuda()
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)

    encoder1 = FreezeEncoder(encoder)
    encoder2 = FreezeEncoder(decoder)

    return encoder1, encoder2
