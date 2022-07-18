import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
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

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Res_GCN_BCP(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 spatial_scale,
                 bn_layer=False):
        super(Res_GCN_BCP, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d
        self.spatial_scale = spatial_scale
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, v):

        batch_size, c_size, h_size, w_size = v.size(0), v.size(1), v.size(2), v.size(3)

        v_3d = v.view(batch_size, c_size, -1)
        v_reshape = F.interpolate(v, size=(h_size//self.spatial_scale, w_size//self.spatial_scale), mode='bilinear')
        v_reshape = v_reshape.view(batch_size, c_size, -1)

        g_v = self.g(v_reshape).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v_reshape).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)

        phi_v = self.phi(v_reshape).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v_reshape.size()[2:])
        y_reshape = F.interpolate(y, size=(h_size*w_size), mode='linear')

        W_y = self.W(y_reshape).mul(0.1)
        W_y = torch.sigmoid(W_y)

        v_star = W_y * v_3d + v_3d
        v_star = v_star.view(batch_size, c_size, h_size, w_size)

        return v_star


class MyBlock(nn.Module):
    def __init__(self, conv, n_feats,
                 kernel_size,bias=True,
                 bn=False, act=nn.ReLU(True),
                 res_scale=1):

        super(MyBlock, self).__init__()
        m = []

        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res


class TRAIN_MODEL(nn.Module):
    def __init__(self,
                 in_ch  = 28,
                 out_ch = 28,
                 noise_mean = 0.,
                 noise_std = 1.,
                 init = 'uniform',
                 conv = default_conv,
                 noise_act = nn.Softplus(),
                 inter_channels = 16,
                 spatial_scale = 4):
        super(TRAIN_MODEL, self).__init__()

        n_resblocks = 16
        n_feats_noise = in_ch
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        noise_act = noise_act

        m_head = [conv(in_ch, n_feats, kernel_size)]

        m_noise_body = [MyBlock(conv, n_feats_noise, kernel_size, act=act, res_scale=1),Res_GCN_BCP(n_feats_noise, inter_channels, spatial_scale)]
        m_noise_body.append(act)
        m_noise_body.append(conv(n_feats_noise, 1, kernel_size,))
        m_noise_body.append(noise_act)

        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale= 1) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [conv(n_feats, out_ch, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.noise_body = nn.Sequential(*m_noise_body)
        self.tail = nn.Sequential(*m_tail)
        self.mean = noise_mean
        self.std = noise_std
        self.init = init

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

        g_phi_m = self.noise_body(input_mask)

        noise_prior = g_phi_m.new(g_phi_m.size()).normal_(mean=self.mean, std=self.std)
        noise_posterior = g_phi_m * noise_prior
        perturb_mask = input_mask + noise_posterior
        perturb_mask_clamp = torch.clamp(perturb_mask, min=self.mean, max=1.)
        x = torch.mul(input_data, perturb_mask_clamp)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x, noise_posterior, noise_prior, perturb_mask

