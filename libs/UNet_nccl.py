import torch
import os
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Dict, List
import copy
import torch.profiler as profiler

def initialize_model_with_pretrained_weights(new_model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path)
    
    new_model_state_dict = new_model.state_dict()
    
    for name, param in new_model_state_dict.items():
        # print(name)
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

def scatter_slices(rank, x, split, device_list):
    if rank == 0:
        B, C, H, W = x.size()
        split_y, split_x = split

        split_H = H // split_y
        split_W = W // split_x
        slice_list = []
        shape_tensor = torch.tensor((B, C, split_H, split_W), device=device_list[rank], dtype=torch.int)
        dist.broadcast(shape_tensor, src=0)
    else:
        shape_tensor = torch.zeros(4, device=device_list[rank], dtype=torch.int)
        dist.broadcast(shape_tensor, src=0)
        B, C, split_H, split_W = shape_tensor.tolist()
        slice_list = None
    if rank == 0:
        for i in range(split_y):
            for j in range(split_x):
                slice = x[:, :, 
                            i*split_H:(i+1)*split_H, 
                            j*split_W:(j+1)*split_W].contiguous()
                slice_list.append(slice)
    local_slice = torch.empty((B, C, split_H, split_W), device=device_list[rank])
    dist.scatter(local_slice, scatter_list=slice_list, src=0)
    return local_slice

def gather_slices(rank, local_slice, split, device_list):
    B, C, split_H, split_W= local_slice.size()
    split_y, split_x = split
    size = split_H * split_y
    if rank == 0:
        gathered_slices = [torch.empty((B, C, split_H, split_W), 
                                    device=device_list[rank]) 
                                    for i in range(len(device_list))]
        dist.barrier()
        dist.gather(local_slice, gather_list=gathered_slices, dst=0)
        output = torch.empty((B, C, size, size), device=device_list[rank])
        for i in range(split_y):
            for j in range(split_x):
                output[:, :, 
                       i*split_H:(i+1)*split_H, 
                       j*split_W:(j+1)*split_W] = gathered_slices[i*split_x + j]
        return output
    else:
        dist.barrier()
        dist.gather(local_slice, gather_list=None, dst=0)
        return None      

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
        self.input = input
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
    

    def halo_exchange(self) -> torch.Tensor:
        padding = self.padding
        input = self.input
        padded_input = F.pad(input, (padding, padding, padding, padding))

        if self.split == 1:
            return padded_input
        split_y, split_x = self.split
        
        row = self.row
        col = self.col
        B, C, H, W = input.size()
        # Create a halo tensor that contains the necessary padding for halo exchange
        halo_size = padding # 1
        halo_tensor = torch.cat([input[:, :, 0, :],
                                 input[:, :, :, 0],
                                 input[:, :, :, -1],
                                 input[:, :, -1, :]]
                                 ,dim=2
                                 ).contiguous()
        halo_list = [halo_tensor.clone() for _ in range(self.world_size)]
        dist.all_to_all(halo_list, halo_list)
        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    halo_list[(row - 1) * split_x + (col - 1)][:, :, -1].reshape(B, C, 1, 1) # top_left
            padded_input[:, :, :1, padding:-padding] = \
                halo_list[(row - 1) * split_x + col][:, :, -W:].reshape(B, C, 1, W) # top
            if col  < split_x - 1:
                padded_input[:, :, :1, -1:] = \
                    halo_list[(row - 1) * split_x + (col + 1)][:, :, -W].reshape(B, C, 1, 1) # top_right
        if col > 0:
            padded_input[:, :, padding:-padding, :1] = \
                halo_list[row * split_x + (col - 1)][:, :, -(H+W):-W].reshape(B, C, H, 1) # left
        if col < split_x - 1:
            padded_input[:, :, padding:-padding, -1:] = \
                halo_list[row * split_x + (col + 1)][:, :, W:(H+W)].reshape(B, C, H, 1) # right
        if row < split_y - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    halo_list[(row + 1) * split_x + (col - 1)][:, :, W].reshape(B, C, 1, 1) # bottom_left
            padded_input[:, :, -1:, padding:-padding] = \
                halo_list[(row + 1) * split_x + col][:, :, :W].reshape(B, C, 1, W) # bottom 
            if col < split_x - 1:      
                padded_input[:, :, -1:, -1:] = \
                    halo_list[(row + 1) * split_x + (col + 1)][:, :, 0].reshape(B, C, 1, 1) # bottom_right
        
        return padded_input

    def conv(self, is_first: bool = False):
        padded_input = self.halo_exchange()

        if is_first:
            self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride = self.stride, padding = 0)
        else:
            self.output = F.conv2d(padded_input, self.weight, stride = self.stride, padding = 0)
        
    def assign_input(self, input: torch.Tensor):
        self.input = input
        self.output = None
        self.avg = None
        self.output_list = [self.input]

    
    def synchronize(self):
        self.output_list.append(self.output)
        self.input = self.lrelu(self.pool(self.output)) if self.output.size(2) > 1 else None
         
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
        self.input = torch.cat([input,self.relu(input_list[-1])], dim=1)
        input_list.pop()
        self.dilate_input(stride=2)
        self.input_list = input_list
        self.output = None
        self.avg = None
        
    def dilate_input(self,stride=2):
        B, C, H, W = self.input.shape

        dilated_input = torch.zeros(B, C, 
                                    H * stride - stride + 1, 
                                    W * stride - stride + 1, 
                                    device=self.device)
        
        dilated_input[:, :, ::stride, ::stride] = self.input
        self.dilated_input = dilated_input

    def halo_exchange(self, kernel_size: int = 4 ) -> torch.Tensor:
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
        halo_tensor = torch.cat([self.dilated_input[:, :, 0, :].flatten(2),
                                       self.dilated_input[:, :, :, 0].flatten(2),
                                       self.dilated_input[:, :, :, -1].flatten(2),
                                       self.dilated_input[:, :, -1, :].flatten(2)]
                                       ,dim=2
                                       )
        halo_list = [halo_tensor] * self.world_size
        dist.all_to_all(halo_list, halo_list)

        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    halo_list[(row - 1) * split_x + (col - 1)][:, :, -1].reshape(B, C, 1, 1) # top_left
            padded_input[:, :, :1, real_padding:-real_padding] = \
                halo_list[(row - 1) * split_x + col][:, :, -W:].reshape(B, C, 1, W) # top
            if col  < split_x - 1:
                padded_input[:, :, :1, -1:] = \
                    halo_list[(row - 1) * split_x + (col + 1)][:, :, -W].reshape(B, C, 1, 1) # top_right
        if col > 0:
            padded_input[:, :, real_padding:-real_padding, :1] = \
                halo_list[row * split_x + (col - 1)][:, :, -(H+W):-W].reshape(B, C, H, 1) # left
        if col < split_x - 1:
            padded_input[:, :, real_padding:-real_padding, -1:] = \
                halo_list[row * split_x + (col + 1)][:, :, W:(H+W)].reshape(B, C, H, 1) # right
        if row < split_y - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    halo_list[(row + 1) * split_x + (col - 1)][:, :, W].reshape(B, C, 1, 1) # bottom_left
            padded_input[:, :, -1:, real_padding:-real_padding] = \
                halo_list[(row + 1) * split_x + col][:, :, :W].reshape(B, C, 1, W) # bottom 
            if col < split_x - 1:      
                padded_input[:, :, -1:, -1:] = \
                    halo_list[(row + 1) * split_x + (col + 1)][:, :, 0].reshape(B, C, 1, 1) # bottom_right
        return padded_input

    def conv(self):
        padded_input = self.halo_exchange()
        self.output = F.conv2d(padded_input, self.weight, stride=1, padding=0)
    
    def conv_0(self):
        padded_input = self.halo_exchange(kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride=1, padding=0)
        self.output = self.norm_0(self.output)

    def conv_last(self)->torch.Tensor:
        padded_input = self.halo_exchange(kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_last_convtr, bias=self.bias_last_convtr, stride=1, padding=0)
        norm_output = self.norm_last(self.output)
        tanh_input = self.tanh(norm_output)
        output = F.conv2d(tanh_input, self.weight_last_conv, bias=self.bias_last_conv, stride=1, padding=0)
        return output


    def synchronize(self,stride=2): 
        self.output = self.relu(self.output)

        if len(self.input_list) > 1 or self.input_list[-1].size(2) > 64:
            self.input = torch.cat([self.output, self.relu(self.input_list[-1])], dim=1) 
            self.input_list.pop()
            self.dilate_input(stride) 

    def synchronize_last(self):
        self.input = torch.cat([self.output, self.input_list[-1]], dim=1)
        # self.input_list.pop()
        self.dilate_input(stride=1)
    
class UNet(nn.Module):
    def __init__(self, kc, inc, ouc, device_list, split):
        super(UNet, self).__init__()
        self.device_list = device_list
        split_y, split_x = split
        self.split = split
        self.world_size = split_x*split_y
        assert len(device_list) >= split_y * split_x, "Device count must match split"
        assert (split_x & (split_x - 1)) == 0, "split_x must be a power of 2 but got {}".format(split_x)
        assert (split_y & (split_y - 1)) == 0, "split_y must be a power of 2 but got {}".format(split_y)
        print("input size should be larger than ", max(split) * 64, "(split * 64)")
        


        self.down_slices : List[DownsampleSlices] = []
        self.up_slices : List[UpsampleSlices] = []
        
        for device in device_list:
            self.down_slices.append(DownsampleSlices(device))
            self.up_slices.append(UpsampleSlices(device,ch1=kc, ch2=inc))
  
        self.down_slice_0 = DownsampleSlices(device_list[0])
        self.up_slices_0 = UpsampleSlices(device_list[0], ch1=kc, ch2=inc)
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
    
    def _setup(self, rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    def _cleanup(self):
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    def _unet_dist(self, rank: int, input: torch.Tensor, local_output: torch.Tensor, result_queue: mp.Queue):
        with torch.no_grad():
            world_size = self.world_size
            down_slice = self.down_slices[rank]
            up_slice = self.up_slices[rank]
            split_y, split_x= self.split
            size = input.size(2) * split_y
            layer_num = 0
            while (1 << layer_num) < size:
                layer_num += 1
            down_slice.assign_input(input)
            down_slice.conv(is_first=True)
            down_slice.synchronize()
            for i in range(layer_num):
                down_slice.conv()
                down_slice.dist_normalize(world_size)
                down_slice.synchronize()
                _, _, temp_h,temp_w = down_slice.input.size()
                if temp_h*split_y <= 64 or min(temp_h, temp_w) <= 4:
                    inter_layer_num = 0
                    while (1 << inter_layer_num) < temp_h*split_y:
                        inter_layer_num += 1
                    break
            inter_input = gather_slices(rank, down_slice.input, self.split, self.device_list)
            if rank == 0:
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

            inter_output = scatter_slices(rank, self.up_slices_0.output, self.split, self.device_list)
            up_slice.assign_input(inter_output, down_slice.output_list)

            for i in range(layer_num - inter_layer_num - 1):
                up_slice.conv()
                up_slice.dist_normalize(world_size)
                if i == layer_num - inter_layer_num - 2:
                    up_slice.synchronize(stride=1)
                else:
                    up_slice.synchronize()

            up_slice.conv_0()
            up_slice.synchronize_last()
            last_conv_output = up_slice.conv_last()

            output = torch.where(last_conv_output >= 0, torch.exp(last_conv_output)-1, -torch.exp(-last_conv_output)+1).to(torch.device('cpu'))

            local_output.copy_(output)

            result_queue.put("done")

    def forward(self, x):        
        x = x.to(self.device_list[0]) 
        shared_output = torch.empty_like(x)
        shared_output.share_memory_()
        mp.spawn(
            self._unet_dist,
            args=(x,shared_output,),
            nprocs=len(self.device_list),
            join=True
        )
        output = shared_output.to(self.device_list[0])
        return output

