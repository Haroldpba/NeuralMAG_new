from pyexpat import model
import torch
import numpy as np
from UNet_nccl_out import UNet as UNet_nccl
from Unet_base import UNet as UNet_base
from Unet_ser import UNet as UNet_ser
from utils import initialize_model_with_pretrained_weights, tensor2rgb, FFT_Hd
from UNetManager import UNetManager
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
def test():
    ckpt = 'model_fs.pt'
    print('Model base loaded from:', ckpt)
    device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    division_shape = (2,2)
    # model_base = UNet_base(kc=48, inc=6, ouc=6).eval().to(torch.device('cuda:3'))
    # initialize_model_with_pretrained_weights(model_base, ckpt)

    model_ser = UNet_ser(kc=48, inc=6, ouc=6, device=torch.device('cuda:0'), split = 2).eval().to(torch.device('cuda:0'))
    model_ser.load_weights(ckpt)

    Unet_nccl_manager = UNetManager(kc=48, inc=6, ouc=6, device_list=device_list, split=2, cpkt=ckpt)


    # inputs_32 = np.load('data32.npy').transpose(2, 0, 1)
    # inputs_32 = torch.from_numpy(inputs_32).unsqueeze(0).to('cuda:0')
    # inputs_256 = np.load('data256.npy').transpose(2, 0, 1)
    # inputs_256 = torch.from_numpy(inputs_256).unsqueeze(0).to('cuda:0')
    inputs_1024 = np.load('data1024.npy').transpose(2, 0, 1)
    inputs_1024 = torch.from_numpy(inputs_1024).unsqueeze(0).to('cuda:0')

    inputs_list = [inputs_1024]
    size_list = [1024]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for size, inputs in zip(size_list, inputs_list):
            # outputs_base = model_base(inputs)[0]
            outputs_ser = model_ser(inputs)[0]
            outputs_nccl = Unet_nccl_manager.predict(inputs)[0]
            # outputs_nccl = outputs_nccl.to(torch.device("cuda:0"))
            print("outputs_ser", outputs_ser[0])
            # print("outputs_base",outputs_base[0])
            print("outputs_nccl",outputs_nccl[0])
            mse_bse = torch.sum((outputs_ser.cpu() - outputs_nccl) ** 2, dim=0).cpu().numpy()
            print("mse_bse",mse_bse)
            print("mse_bse mean",mse_bse.mean())
if __name__ == '__main__':
    test()
    

# fft
# torch.compile(model_nccl
# work_process, model在process中初始化, 同样只初始化一次