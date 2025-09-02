from pyexpat import model
import torch
import numpy as np
from libs.UNet_base import UNet as UNet_base
# from Unet_ser import UNet as UNet_ser
from libs.UNet_ser import UNet as UNet_ser

# from utils import initialize_model_with_pretrained_weights, tensor2rgb, FFT_Hd
from libs.UNetManager import UNetManager
import multiprocessing as mp

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

def test():
    ckpt = '../cpkt/k48/model.pt'
    print('Model base loaded from:', ckpt)
    device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:5']
    device = torch.device('cuda:0')
    model_base = UNet_base(kc=48, inc=6, ouc=6).eval().to(torch.device('cuda:0'))
    initialize_model_with_pretrained_weights(model_base, ckpt)

    # model_ser = UNet_ser(kc=48, inc=6, ouc=6, device=device, split =(2,2)).eval().to(device)
    # model_ser.load_weights(ckpt)

    Unet_nccl_manager = UNetManager(kc=48, inc=6, ouc=6, device_list=device_list, split=(2,2), cpkt=ckpt, input_shape=(1, 6, 1024, 1024))


    # inputs_32 = np.load('data32.npy').transpose(2, 0, 1)
    # inputs_32 = torch.from_numpy(inputs_32).unsqueeze(0).to('cuda:0')
    # inputs_256 = np.load('data256.npy').transpose(2, 0, 1)
    # inputs_256 = torch.from_numpy(inputs_256).unsqueeze(0).to('cuda:0')
    inputs_1024 = np.load('data1024.npy').transpose(2, 0, 1)
    inputs_1024 = torch.from_numpy(inputs_1024).unsqueeze(0).to(torch.device('cpu'))

    inputs_list = [inputs_1024]
    size_list = [1024]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for size, inputs in zip(size_list, inputs_list):
            outputs_base = model_base(inputs.to(device))[0]
            # outputs_ser = model_ser(inputs)[0]
            outputs_nccl = Unet_nccl_manager.predict(inputs)[0]
            print("outputs_base",outputs_base[0])
            # print("outputs_ser", outputs_ser[0])
            print("outputs_nccl",outputs_nccl[0])
            mse_bse = torch.sum((outputs_nccl - outputs_base.cpu()) ** 2, dim=0).cpu().numpy()
            print("mse_bse",mse_bse)
            print("mse_bse mean",mse_bse.mean())
        # for i in range(100):
        #     print("start predict", i)
        #     outputs_nccl = Unet_nccl_manager.predict(inputs_1024)
    Unet_nccl_manager.close()
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    test()
    
