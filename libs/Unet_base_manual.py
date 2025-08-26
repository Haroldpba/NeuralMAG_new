import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
def initialize_model_with_pretrained_weights(new_model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path)
    
    new_model_state_dict = new_model.state_dict()
    
    for name, param in new_model_state_dict.items():
        print(name)
        if name in pretrained_weights:
            new_model_state_dict[name].copy_(pretrained_weights[name])
        if name == 'weight_s0':
            new_model_state_dict[name].copy_(pretrained_weights['s0.weight'])
        if name == 'bias_s0':
            new_model_state_dict[name].copy_(pretrained_weights['s0.bias'])
        if name == 'weight_up0':
            new_model_state_dict[name].copy_(pretrained_weights['up0.up.1.weight'])
        if name == 'bias_up0':
            new_model_state_dict[name].copy_(pretrained_weights['up0.up.1.bias'])
        if name == 'weight_last_tr':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.0.weight'])
        if name == 'bias_last_tr':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.0.bias'])
        if name == 'weight_last_c':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.3.weight'])
        if name == 'bias_last_c':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.3.bias'])
        if 'norm' in name:
            if 'up0' in name:
                new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm', 'up.2')])
            if 'last' in name:
                new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm', '1')])
        
    new_model.load_state_dict(new_model_state_dict)

def manual_convtranspose2d(input, weight, stride: int, padding: int):
    output = F.conv_transpose2d(input, weight, stride=stride, padding=padding)
    return output
def manual_conv2d(input, weight, stride: int, padding: int):
    output = F.conv2d(input, weight, stride=stride, padding=padding)
    return output
    
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(downsample, self).__init__()
        self.stride = s
        self.padding = p
        self.pool = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.drop = nn.Dropout(drop_out) if drop_out>0 else nn.Identity()

    def forward(self, x, weight):
        pool_x = self.pool(x)
        lrelu_x = self.lrelu(pool_x)
        conv_x = manual_conv2d(lrelu_x, weight, stride=self.stride, padding=self.padding)
        norm_x = (conv_x-conv_x.mean(dim=(1, 2, 3), keepdim=True))/(conv_x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
        drop_x = self.drop(norm_x)
        return drop_x
    
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample, self).__init__()
        self.stride = s
        self.padding = p
        self.relu = nn.ReLU(True)

    def forward(self, x, weight):
        relu_x = self.relu(x)
        convt_x = manual_convtranspose2d(relu_x, weight, stride=self.stride, padding=self.padding)
        norm_x = (convt_x-convt_x.mean(dim=(1, 2, 3), keepdim=True))/(convt_x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
        return norm_x
        
class upsample_0(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample_0, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kr, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_out) if drop_out>0 else nn.Identity()
            )
        
    def forward(self, x):
        return self.up(x)

# 32*32
class UNet(nn.Module):
    def __init__(self, kc=16, inc=3, ouc=3):
        super(UNet, self).__init__()
        self.s0 = nn.Conv2d(  inc,   1*kc,  3,1,1)
        self.s = downsample( 1*kc,  1*kc,  3,1,1, drop_out=0.0)

        self.weight_s  = nn.Parameter(torch.zeros(1*kc, 1*kc, 3, 3))
        self.weight_up = nn.Parameter(torch.zeros(2*kc, 1*kc, 4, 4))

        self.up = upsample( 2*kc, 1*kc, 4,2,1, drop_out=0.0)
        self.up0 = upsample_0( 2*kc, 1*kc, 3,1,1, drop_out=0.0)

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(kc+inc, 1*kc, 3,1,1),
            nn.BatchNorm2d(1*kc),
            nn.Tanh(),
            nn.Conv2d(1*kc, ouc, 1,1,0),
        )
        self.is_half = False
        print("c_gn_test")
        self.init_weight()
    def init_weight(self):
        nn.init.kaiming_normal_(self.weight_s, mode='fan_out')
        nn.init.kaiming_normal_(self.weight_up, mode='fan_in')
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
    def half_precision(self):
        self.half()
        self.is_half = True
    def forward(self, x):
        batch = x.size(0)
        size = x.size(2)
        layer_num =size.bit_length() - 1
        if self.is_half:
            x = x.half()
            layer_out: List[torch.Tensor] = [torch.empty((batch, 48, int(size/pow(2,i)), int(size/pow(2,i))),device=torch.device("cuda:0"), dtype=torch.float16) for i in range(layer_num+2)]
        else:
            layer_out: List[torch.Tensor] = [torch.empty((batch, 48, int(size/pow(2,i)), int(size/pow(2,i))),device=torch.device("cuda:0")) for i in range(layer_num+2)]

        layer_out[0] = self.s0(x)
        for i in range(layer_num):
            layer_out[i+1] = self.s(layer_out[i], self.weight_s)

        layer_out[layer_num+1] = layer_out[layer_num]
        for i in range(layer_num):
            layer_out[layer_num-i] = self.up(torch.cat([layer_out[layer_num-i+1], layer_out[layer_num-i]], dim=1), self.weight_up)
        
        up_0 = self.up0(torch.cat([layer_out[1],layer_out[0]], dim=1))
        out  = self.last_Conv(torch.cat([up_0,x],dim=1))
        return torch.where(out >= 0, torch.exp(out)-1, -torch.exp(-out)+1)
