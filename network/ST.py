import torch
import torch.nn as nn
_NORM_BONE = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def conv_block(in_planes, out_planes, the_kernel=3, the_stride=1, the_padding=1, flag_norm=False, flag_norm_act=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=the_kernel, stride=the_stride, padding=the_padding)
    activation = nn.ReLU(inplace=True)
    norm = nn.BatchNorm2d(out_planes)
    if flag_norm:
        return nn.Sequential(conv,norm,activation) if flag_norm_act else nn.Sequential(conv,activation,norm)
    else:
        return nn.Sequential(conv,activation)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, res=True):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.res = res

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        if self.res:
            res += x
        else:
            pass
        return res

class ST_MODEL(nn.Module):
    def __init__(self, in_ch=28,
                 out_ch=28,
                 noise_mean=0.006,
                 noise_std=0.3,
                 init='uniform',
                 conv=default_conv,
                 noise_act = nn.Softplus()):
        super(ST_MODEL, self).__init__()

        noise_resblocks = 1
        n_resblocks = 16
        n_feats_noise = in_ch
        n_feats = 64
        kernel_size = 3
        scale = 1
        act = nn.ReLU(True)
        noise_act = noise_act

        # define head module
        m_head = [conv(in_ch, n_feats, kernel_size)]


        # define noise body module
        m_noise_body = [
            ResBlock(
                conv, n_feats_noise, kernel_size, act=act, res_scale=1, res=False
            ) for _ in range(noise_resblocks)
        ]
        m_noise_body.append(act)
        m_noise_body.append(conv(n_feats_noise, 1, kernel_size,))
        m_noise_body.append(noise_act)

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale= 1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [conv(n_feats, out_ch, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.noise_body = nn.Sequential(*m_noise_body)
        self.tail = nn.Sequential(*m_tail)

        self.mean = noise_mean
        self.std = noise_std
        self.check_grad = None
        self.init = init

        # initialization
        for name, params in self.noise_body.named_parameters():
            if self.init == 'uniform':
                nn.init.uniform_(params, -0.2, 0.2)
            elif self.init == 'normal':
                nn.init.normal_(params, 0.5, 0.01)
            elif self.init == 'xavier_uniform':
                if 'bias' in name:
                    nn.init.constant_(params, 0.)
                else:
                    nn.init.xavier_uniform_(params)
            else:
                print('Invalid init type, should be either [uniform], [normal], [xavier_uniform].')
                raise ValueError


    def forward(self, input_data, input_mask):

        sigma = self.noise_body(input_mask)

        noise = sigma.new(sigma.size()).normal_(mean=self.mean, std=self.std)
        perturb_mask = input_mask + sigma * noise

        perturb_mask_clamp = torch.clamp(perturb_mask, min=self.mean, max=1.)

        x = torch.mul(input_data, perturb_mask_clamp)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x, sigma

