from pyexpat import model
from requests import get
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
        norm_x = (conv_x-conv_x.mean(dim=(1, 2, 3), keepdim=True))/(conv_x.std(dim=(1, 2, 3), unbiased=False, keepdim=True) + 1e-5)
        drop_x = self.drop(norm_x)
        return drop_x
        # return self.drop(conv_x)
    
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample, self).__init__()
        self.stride = s
        self.padding = p
        self.relu = nn.ReLU(True)

    def forward(self, x, weight):
        relu_x = self.relu(x)
        convt_x = manual_convtranspose2d(relu_x, weight, stride=self.stride, padding=self.padding)
        norm_x = (convt_x-convt_x.mean(dim=(1, 2, 3), keepdim=True))/(convt_x.std(dim=(1, 2, 3), unbiased=False, keepdim=True) + 1e-5)
        return norm_x
        # return convt_x
        
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
        self.input_size = None
        print("trt")
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
    def set_input_size(self, input_size):
        if not isinstance(input_size, tuple) or len(input_size) != 4:
            raise ValueError("Input size must be a tuple of (batch_size, channels, height, width).")
        self.input_size = input_size
        batch = self.input_size[0]
        size = self.input_size[2]
        self.layer_num = int(torch.log2(torch.tensor(size, dtype=torch.float32)).item())
        print(f"layer_num: {self.layer_num}")

        dtype = torch.float16 if self.is_half else torch.float32
        device = torch.device("cuda:0")

        for i in range(11):
            layer_size = size // (2 ** i)
            buf = torch.empty((batch, 96, layer_size, layer_size), dtype=dtype, device=device)
            self.register_buffer(f"layerout_{i}", buf, persistent=False)
    def get_layerout(self, idx: int) -> torch.Tensor:
        if idx == 0:
            return self.layerout_0
        elif idx == 1:
            return self.layerout_1
        elif idx == 2:
            return self.layerout_2
        elif idx == 3:
            return self.layerout_3
        elif idx == 4:
            return self.layerout_4
        elif idx == 5:
            return self.layerout_5
        elif idx == 6:
            return self.layerout_6
        elif idx == 7:
            return self.layerout_7
        elif idx == 8:
            return self.layerout_8
        elif idx == 9:
            return self.layerout_9
        elif idx == 10:
            return self.layerout_10
        else:
            raise ValueError(f"Invalid idx: {idx}")

    def forward(self, x):
        # 获取x.size(2)中2的因子数量
        if self.input_size is None:
            raise ValueError("Input size is not set. Please set input size before forward pass.")
        layer_num = self.layer_num
        layer_out0 = self.get_layerout(0)
        layer_out0[:,48:,:,:].copy_(self.s0(x))
        for i in range(layer_num):
            input_buf = self.get_layerout(i)
            output_buf = self.get_layerout(i + 1)
            out = self.s(input_buf[:,48:,:,:], self.weight_s)
            output_buf[:,48:,:,:].copy_(out)
        layer_out_last =self.get_layerout(layer_num)
        layer_out_last[:,0:48,:,:].copy_(layer_out_last[:,48:,:,:])

        for i in range(layer_num):
            input_buf = self.get_layerout(layer_num-i)
            output_buf = self.get_layerout(layer_num-i-1)
            output_buf[:,0:48,:,:].copy_(self.up(input_buf, self.weight_up))
        layer_out_0 = self.get_layerout(0)
        up_0 = self.up0(layer_out_0)
        out  = self.last_Conv(torch.cat([up_0,x],dim=1))
        return torch.where(out >= 0, torch.exp(out)-1, -torch.exp(-out)+1)
    