import re
from numpy import var
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Dict, List
import copy

def initialize_model_with_pretrained_weights(new_model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path)
    
    new_model_state_dict = new_model.state_dict()
    
    for name, param in new_model_state_dict.items():
        print(name)
        if name in pretrained_weights:
            new_model_state_dict[name].copy_(pretrained_weights[name])
            continue
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
        if 'norm1' in name:
            new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm1', 'up0.up.2')])
        if 'norm2' in name:
            new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm2', 'last_Conv.1')])

    new_model.load_state_dict(new_model_state_dict)

def scatter_slices(output, up_slices, down_slices, split):
    size = output.size(2)
    split_y, split_x = split
    for i in range(split_y):
        for j in range(split_x):
            idx = i * split_x + j
            inter_output = output[:, :,
                                i*size//split_y:(i+1)*size//split_y, 
                                j*size//split_x:(j+1)*size//split_x].contiguous()
            up_slices[idx].assign_input(inter_output, down_slices[idx].output_list)

def gather_slices(down_slices, split):
    batch_size, channels, height, width = down_slices[0].input.size()
    split_y, split_x = split
    size = height * split_y
    output = torch.empty((batch_size, channels, size, size))
    for i in range(split_y):
        for j in range(split_x):
            output[:, :, 
                    i*height:(i+1)*height, 
                    j*width:(j+1)*width] = down_slices[i*split_x + j].input
    return output

class Slices:
    def __init__(
        self,
        device: torch.device,
    ):
        self.row = None
        self.col = None
        self.split = None

        self.output = None
        self.avg = None
        self.var = None
        self.device = device
        self.world_size = None
    def set_neighbor(self, direction: str, neighbor: Optional['Slices']):
        assert direction in self.neighbors, f"Invalid direction: {direction}"
        self.neighbors[direction] = neighbor
    
    def get_average(self) -> torch.Tensor:
        avg = torch.mean(self.output, dim=(1, 2, 3), keepdim=True)
        self.avg = avg
        return avg
    
    def get_variance(self) -> torch.Tensor:
        var = torch.mean(torch.square(self.output - self.avg), dim=(1, 2, 3), keepdim=True)
        return var

    def dist_normalize(self, world_size: int):
        avg = self.get_average()
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        avg /= world_size

        var = self.get_variance()
        dist.all_reduce(var, op=dist.ReduceOp.SUM)
        var /= world_size

        self.output = (self.output - avg) / torch.sqrt(var + 1e-5)

    def self_normalize(self):
        avg = torch.mean(self.output, dim=(1, 2, 3), keepdim=True)
        var = torch.mean(torch.square(self.output - avg), dim=(1, 2, 3), keepdim=True)
        self.output = (self.output - avg) / torch.sqrt(var + 1e-5)

    def assign_input(self, input: torch.Tensor):
        self.input = input.to(self.device)  # 确保张量移动到对应的 GPU
        self.output = None
        self.avg = None

class DownsampleSlices(Slices):
    def __init__(
        self,
        device: torch.device,
        weight: Optional[torch.Tensor] = None,
        weight_0: Optional[torch.Tensor] = None,
        bias_0: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 1
    ):
        super(DownsampleSlices, self).__init__(device)
        self.output_list = []

        self.weight = weight.to(device) if weight is not None else None
        self.weight_0 = weight_0.to(device) if weight_0 is not None else None
        self.bias_0 = bias_0.to(device) if bias_0 is not None else None
        self.stride = stride
        self.padding = padding

        self.pool = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
    

    def halo_exchange(self, neighbors) -> torch.Tensor:
        padding = self.padding
        input = self.input.to(self.device)
        padded_input = F.pad(input, (padding, padding, padding, padding))

        if self.split == 1:
            return padded_input
        split_y, split_x = self.split
        
        row = self.row
        col = self.col
        B, C, H, W = input.size()
        # Create a halo tensor that contains the necessary padding for halo exchange
        halo_size = padding # 1

        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    neighbors[(row - 1) * split_x + (col - 1)].halo_tensor[3][:, :, -1].reshape(B, C, 1, 1).to(self.device) # top_left
            padded_input[:, :, :1, padding:-padding] = \
                neighbors[(row - 1) * split_x + col].halo_tensor[3][:, :, :].reshape(B, C, 1, W).to(self.device) # top
            if col  < split_x - 1:
                padded_input[:, :, :1, -1:] = \
                    neighbors[(row - 1) * split_x + (col + 1)].halo_tensor[3][:, :, 0].reshape(B, C, 1, 1).to(self.device) # top_right
        if col > 0:
            padded_input[:, :, padding:-padding, :1] = \
                neighbors[row * split_x + (col - 1)].halo_tensor[2][:, :, :].reshape(B, C, H, 1).to(self.device) # left
        if col < split_x - 1:
            padded_input[:, :, padding:-padding, -1:] = \
                neighbors[row * split_x + (col + 1)].halo_tensor[1][:, :, :].reshape(B, C, H, 1).to(self.device) # right
        if row < split_y - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    neighbors[(row + 1) * split_x + (col - 1)].halo_tensor[0][:, :, -1].reshape(B, C, 1, 1).to(self.device) # bottom_left
            padded_input[:, :, -1:, padding:-padding] = \
                neighbors[(row + 1) * split_x + col].halo_tensor[0][:, :, :].reshape(B, C, 1, W).to(self.device) # bottom 
            if col < split_x - 1:      
                padded_input[:, :, -1:, -1:] = \
                    neighbors[(row + 1) * split_x + (col + 1)].halo_tensor[0][:, :, 0].reshape(B, C, 1, 1).to(self.device) # bottom_right
        
        return padded_input

    def conv(self, neighbors=None, is_first: bool = False):
        padded_input = self.halo_exchange(neighbors)

        if is_first:
            self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride = self.stride, padding = 0)
        else:
            self.output = F.conv2d(padded_input, self.weight, stride = self.stride, padding = 0)
    def assign_input(self, input: torch.Tensor):
        self.input = input
        self.output = None
        self.avg = None
        self.output_list = [self.input]
        self.halo_tensor = [input[:, :, 0, :],
                            input[:, :, :, 0],
                            input[:, :, :, -1],
                            input[:, :, -1, :]]

    def synchronize(self):
        self.output_list.append(self.output.to(torch.device('cpu')))
        if self.output.size(2) <= 1:
            return

        self.input = self.lrelu(self.pool(self.output)) 
        self.input = self.input.to(torch.device('cpu')) 
        self.halo_tensor = [self.input[:, :, 0, :],
                            self.input[:, :, :, 0],
                            self.input[:, :, :, -1],
                            self.input[:, :, -1, :]]

class UpsampleSlices(Slices):
    def __init__(
        self,
        device: torch.device,
        weight: Optional[torch.Tensor] = None,
        weight_0: Optional[torch.Tensor] = None,
        bias_0: Optional[torch.Tensor] = None,
        weight_last_convtr: Optional[torch.Tensor] = None,
        bias_last_convtr: Optional[torch.Tensor] = None,
        weight_last_conv: Optional[torch.Tensor] = None,
        bias_last_conv: Optional[torch.Tensor] = None,
        stride: int = 2,
        padding: int = 1,
        ch1: int = 48,
        ch2: int = 6
    ):
        super(UpsampleSlices, self).__init__(device)

        self.weight = weight.to(device) if weight is not None else None
        self.weight_0 = weight_0.to(device) if weight_0 is not None else None
        self.bias_0 = bias_0.to(device) if bias_0 is not None else None
        self.weight_last_convtr = weight_last_convtr.to(device) if weight_last_convtr is not None else None
        self.bias_last_convtr = bias_last_convtr.to(device) if bias_last_convtr is not None else None
        self.weight_last_conv = weight_last_conv.to(device) if weight_last_conv is not None else None
        self.bias_last_conv = bias_last_conv.to(device) if bias_last_conv is not None else None
        self.stride = stride
        self.padding = padding

        self.relu = nn.ReLU(True)
        self.norm_0 = nn.BatchNorm2d(ch1)
        self.norm_last = nn.BatchNorm2d(ch2)
        self.tanh = nn.Tanh()
    
    def assign_input(self, input:torch.Tensor, input_list: List[torch.Tensor]):
        input = input.to(self.device) 
        self.input = torch.cat([input,self.relu(input_list[-1].to(self.device))], dim=1)
        input_list.pop()
        self.dilate_input(stride=2)
        self.input_list = input_list
        self.output = None
        self.avg = None
        self.halo_tensor = [self.dilated_input[:, :, 0, :],
                            self.dilated_input[:, :, :, 0],
                            self.dilated_input[:, :, :, -1],
                            self.dilated_input[:, :, -1, :]]
        
    def dilate_input(self,stride=2):
        B, C, H, W = self.input.shape

        dilated_input = torch.zeros(B, C, 
                                    H * stride - stride + 1, 
                                    W * stride - stride + 1, 
                                    device=self.device)
        
        dilated_input[:, :, ::stride, ::stride] = self.input
        self.dilated_input = dilated_input

    def halo_exchange(self, neighbors, kernel_size: int = 4 ) -> torch.Tensor:
        real_padding = kernel_size - 2
        padded_input = F.pad(self.dilated_input, (real_padding, real_padding, real_padding, real_padding))
        if self.split == 1:
            return padded_input
        split_y, split_x = self.split
        
        row = self.row
        col = self.col
        B, C, H, W = self.dilated_input.size()
        # Create a halo tensor that contains the necessary padding for halo exchange
        halo_size = 1

        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    neighbors[(row - 1) * split_x + (col - 1)].halo_tensor[3][:, :, -1].reshape(B, C, 1, 1) # top_left
            padded_input[:, :, :1, real_padding:-real_padding] = \
                neighbors[(row - 1) * split_x + col].halo_tensor[3][:, :, :].reshape(B, C, 1, W) # top
            if col  < split_x - 1:
                padded_input[:, :, :1, -1:] = \
                    neighbors[(row - 1) * split_x + (col + 1)].halo_tensor[3][:, :, 0].reshape(B, C, 1, 1) # top_right
        if col > 0:
            padded_input[:, :, real_padding:-real_padding, :1] = \
                neighbors[row * split_x + (col - 1)].halo_tensor[2][:, :, :].reshape(B, C, H, 1) # left
        if col < split_x - 1:
            padded_input[:, :, real_padding:-real_padding, -1:] = \
                neighbors[row * split_x + (col + 1)].halo_tensor[1][:, :, :].reshape(B, C, H, 1) # right
        if row < split_y - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    neighbors[(row + 1) * split_x + (col - 1)].halo_tensor[0][:, :, -1].reshape(B, C, 1, 1) # bottom_left
            padded_input[:, :, -1:, real_padding:-real_padding] = \
                neighbors[(row + 1) * split_x + col].halo_tensor[0][:, :, :].reshape(B, C, 1, W) # bottom 
            if col < split_x - 1:
                padded_input[:, :, -1:, -1:] = \
                    neighbors[(row + 1) * split_x + (col + 1)].halo_tensor[0][:, :, 0].reshape(B, C, 1, 1) # bottom_right
        return padded_input

    def conv(self, neighbors=None):
        padded_input = self.halo_exchange(neighbors)
        self.output = F.conv2d(padded_input, self.weight, stride=1, padding=0)
    
    def conv_0(self, neighbors):
        padded_input = self.halo_exchange(neighbors, kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride=1, padding=0)
        self.output = self.norm_0(self.output)

    def conv_last(self, neighbors)->torch.Tensor:
        padded_input = self.halo_exchange(neighbors, kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_last_convtr, bias=self.bias_last_convtr, stride=1, padding=0)
        norm_output = self.norm_last(self.output)
        tanh_input = self.tanh(norm_output)
        output = F.conv2d(tanh_input, self.weight_last_conv, bias=self.bias_last_conv, stride=1, padding=0)
        return output


    def synchronize(self,stride=2): 
        self.output = self.relu(self.output)

        if len(self.input_list) > 1 or self.input_list[-1].size(2) > 64 or self.input_list[-1].size(3) > 64:
            self.input = torch.cat([self.output, self.relu(self.input_list[-1].to(self.device))], dim=1) 
            self.input_list.pop()
            self.dilate_input(stride) 
            self.halo_tensor = [self.dilated_input[:, :, 0, :],  
                                self.dilated_input[:, :, :, 0],
                                self.dilated_input[:, :, :, -1],
                                self.dilated_input[:, :, -1, :]]
    def synchronize_last(self):
        self.input = torch.cat([self.output, self.input_list[-1].to(self.device)], dim=1)
        # self.input_list.pop()
        self.dilate_input(stride=1)
        self.halo_tensor = [self.dilated_input[:, :, 0, :],  
                    self.dilated_input[:, :, :, 0],
                    self.dilated_input[:, :, :, -1],
                    self.dilated_input[:, :, -1, :]]
    
class UNet(nn.Module):
    def __init__(self, kc, inc, ouc, device, split):
        super(UNet, self).__init__()
        self.device = device
        split_x, split_y = split
        assert (split_x & (split_x - 1)) == 0, "split_x must be a power of 2 but got {}".format(split_x)
        assert (split_y & (split_y - 1)) == 0, "split_y must be a power of 2 but got {}".format(split_y)
        print("input size should be larger than ", max(split) * 64, "(split * 64)")
        
        self.split = split
        self.world_size = split[0]*split[1]

        self.down_slices : List[DownsampleSlices] = []
        self.up_slices : List[UpsampleSlices] = []
        
        for _ in range(self.world_size):
            self.down_slices.append(DownsampleSlices(device))
            self.up_slices.append(UpsampleSlices(device,ch1=kc, ch2=inc))
  
        self.down_slice_0 = DownsampleSlices(device)
        self.up_slices_0 = UpsampleSlices(device, ch1=kc, ch2=inc)
        self._assign_location()

        self.weight_s0 = nn.Parameter(torch.zeros(1*kc, inc, 3, 3))
        self.bias_s0 = nn.Parameter(torch.zeros(1*kc))
        
        self.weight_s  = nn.Parameter(torch.zeros(1*kc, 1*kc, 3, 3))
        
        self.weight_up = nn.Parameter(torch.zeros(2*kc, 1*kc, 4, 4))

        self.weight_up0 = nn.Parameter(torch.zeros(2*kc, 1*kc, 3, 3))
        self.bias_up0 = nn.Parameter(torch.zeros(1*kc))
        self.norm1 = nn.BatchNorm2d(kc)
        
        self.weight_last_tr = nn.Parameter(torch.zeros(kc+inc, 1*kc, 3, 3))
        self.bias_last_tr = nn.Parameter(torch.zeros(1*kc))
        self.weight_last_c = nn.Parameter(torch.zeros(ouc, 1*kc, 1, 1))
        self.bias_last_c = nn.Parameter(torch.zeros(ouc))
        self.norm2 = nn.BatchNorm2d(kc)
    def _assign_location(self):
        for idx in range(self.world_size):
            row = idx // self.split[1]
            col = idx % self.split[1]

            self.down_slices[idx].row = row
            self.down_slices[idx].col = col
            self.down_slices[idx].split = self.split
            self.down_slices[idx].world_size = self.world_size
            self.up_slices[idx].row = row
            self.up_slices[idx].col = col
            self.up_slices[idx].split = self.split
            self.up_slices[idx].world_size = self.world_size
            
        self.down_slice_0.split = 1
        self.up_slices_0.split = 1
    
    def load_weights(self,cpkt):
        initialize_model_with_pretrained_weights(self, cpkt)
        self.down_slice_0.weight = self.weight_s.to(self.down_slice_0.device).detach().clone()
        for down_slice in self.down_slices:
            down_slice.weight = self.weight_s.to(down_slice.device).detach().clone()
            down_slice.weight_0 = self.weight_s0.to(down_slice.device).detach().clone()
            down_slice.bias_0 = self.bias_s0.to(down_slice.device).detach().clone()

        weight_up = self.weight_up.flip(2).flip(3).transpose(1, 0)
        weight_up0 = self.weight_up0.flip(2).flip(3).transpose(1, 0)
        weight_last_tr = self.weight_last_tr.flip(2).flip(3).transpose(1, 0)
        
        self.up_slices_0.weight = weight_up.to(self.up_slices_0.device).detach().clone()
        for up_slice in self.up_slices:
            up_slice.weight = weight_up.to(up_slice.device).detach().clone()
            up_slice.weight_0 = weight_up0.to(up_slice.device).detach().clone()
            up_slice.bias_0 = self.bias_up0.to(up_slice.device).detach().clone()
            up_slice.weight_last_convtr = weight_last_tr.to(up_slice.device).detach().clone()
            up_slice.bias_last_convtr = self.bias_last_tr.to(up_slice.device).detach().clone()
            up_slice.weight_last_conv = self.weight_last_c.to(up_slice.device).detach().clone()
            up_slice.bias_last_conv = self.bias_last_c.to(up_slice.device).detach().clone()
            up_slice.norm_0 =  copy.deepcopy(self.norm1).to(up_slice.device)
            up_slice.norm_last = copy.deepcopy(self.norm2).to(up_slice.device)
            up_slice.norm_0.eval()
            up_slice.norm_last.eval()

    def down_layer_forward(self, is_first = False):
        if is_first:
            for i, down_slice in enumerate(self.down_slices):
                down_slice.conv(self.down_slices,is_first)
            for down_slice in self.down_slices:
                down_slice.synchronize()
        else:
            avg = 0
            for down_slice in self.down_slices:
                down_slice.conv(self.down_slices,is_first)
                avg += down_slice.get_average()

            avg /= self.world_size
            var = 0
            for down_slice in self.down_slices:
                down_slice.avg = avg
                var += down_slice.get_variance()
            var /= self.world_size

            for i,down_slice in enumerate(self.down_slices):
                down_slice.output = (down_slice.output - avg) / torch.sqrt(var + 1e-5)
                down_slice.synchronize()

    def up_layer_forward(self, stride = 2):
        avg = 0
        for up_slice in self.up_slices:
            up_slice.conv(self.up_slices)
            avg += up_slice.get_average()
        avg /= self.world_size
        var = 0
        for up_slice in self.up_slices:
            up_slice.avg = avg
            var += up_slice.get_variance()
        var /= self.world_size
        for i,up_slice in enumerate(self.up_slices):
            up_slice.output = (up_slice.output - avg) / torch.sqrt(var + 1e-5)
            up_slice.synchronize(stride)

    def forward(self, x):       
        output = torch.empty_like(x, device=torch.device('cpu')) 
        with torch.no_grad():
            split_y, split_x= self.split
            size = x.size(2)
            layer_num = 0
            while (1 << layer_num) < size:
                layer_num += 1
            for idx in range(split_y*split_x):
                i = idx // split_x
                j = idx % split_x

                local_input = x[:, :,
                                i*size//split_y:(i+1)*size//split_y, 
                                j*size//split_x:(j+1)*size//split_x].contiguous()
                self.down_slices[idx].assign_input(local_input)

            self.down_layer_forward(is_first=True)
            
            for i in range(layer_num):
                self.down_layer_forward()

                _, _, temp_h,temp_w = self.down_slices[0].input.size()
                if temp_h*split_y <= 64 or min(temp_h, temp_w) <= 4:
                    inter_layer_num = 0
                    while (1 << inter_layer_num) < temp_h*split_y:
                        inter_layer_num += 1
                    break
            inter_input = gather_slices(self.down_slices, self.split)

            self.down_slice_0.assign_input(inter_input)
            for i in range(inter_layer_num + 1):
                self.down_slice_0.conv()
                self.down_slice_0.self_normalize()
                self.down_slice_0.synchronize()

            self.up_slices_0.assign_input(self.down_slice_0.output, self.down_slice_0.output_list)

            for i in range(inter_layer_num + 1):
                self.up_slices_0.conv()
                self.up_slices_0.self_normalize()
                self.up_slices_0.synchronize()

            scatter_slices(self.up_slices_0.output, self.up_slices, self.down_slices, self.split)

            for i in range(layer_num - inter_layer_num - 1):
                if i == layer_num - inter_layer_num - 2:
                    stride=1
                else:
                    stride=2
                self.up_layer_forward(stride)

            for i,up_slice in enumerate(self.up_slices):
                up_slice.conv_0(self.up_slices)
            for up_slice in self.up_slices:
                up_slice.synchronize_last()
            for idx, up_slice in enumerate(self.up_slices):
                i = idx // split_x
                j = idx % split_x
                last_conv_output = up_slice.conv_last(self.up_slices)
                local_output = torch.where(last_conv_output >= 0, torch.exp(last_conv_output)-1, -torch.exp(-last_conv_output)+1).to(torch.device('cpu'))
                output[:, :,
                        i*size//split_y:(i+1)*size//split_y, 
                        j*size//split_x:(j+1)*size//split_x] = local_output
            return output

