# -*- coding: utf-8 -*-

from ast import arg
import numpy as np
import time
import argparse
import torch
import torch.profiler as profiler
from libs.misc import Culist, spin_prepare, create_trt_model
import libs.MAG2305_halo as MAG2305
# from UNet_nccl import UNet
from UNetManager_test import UNetManager
import multiprocessing as mp


def load_unet_model(args, device_list):
    # load Unet Model
    inch = args.layers*3
    ckpt = 'model_fs.pt'
    model = UNetManager(kc=48, inc=6, ouc=6, device_list=device_list, split=(2,2), cpkt=ckpt,input_shape=(1, 6, args.w, args.w))

    # model = UNet(kc=48, inc=6, ouc=6, device_list=device_list, split= 2).eval().to(torch.device('cuda:0'))
    # model.load_weights(ckpt)
    # Creat trt model
    # if args.trt=='True':
    #     model = create_trt_model(model, inch, args.w, torch.float16, device)
    #     print('Unet model loaded with TensorRT')
    # else:
    #     print('Unet model loaded')
    MAG2305.load_model(model)


def initialize_models(args, device_list):
    #load Unet model
    load_unet_model(args, device_list)

    #Initialize MAG2305 models.
    # film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
    #                         Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
    #                         device="cuda:" + str(args.gpu)
    #                         )
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cpu"
                            )
    print('Creating {} layer models \n'.format(args.layers))

    # spin initialization cases
    spin_split = np.random.randint(low=2, high=32)
    rand_seed = np.random.randint(low=0, high=100000)
    spin = spin_prepare(spin_split, film2, rand_seed)
    film2.SpinInit(spin)
    print('spin shape',film2.Spin.shape)

    return film2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='GPU IDs, e.g. --gpu 0 1 2')
    parser.add_argument('--krn',        type=int,   default=48,        help='unet first layer kernels (default: 16)')
    parser.add_argument('--trt',        type=str,   default='False',   help='unet with tensorRT (default: False)')
    parser.add_argument('--profile',        type=str,   default='False',  help='unet with profiler (default: False)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,0,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--n_loop',     type=int,    default=100,       help='loop number (default: 100)')
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)

    torch.set_default_device('cpu')
    device_list = [f'cuda:{i}' for i in args.gpu]
    
    # Initialize MAG models, prepare films and load Unet model.
    film2 = initialize_models(args, device_list)

    #########################
    #  NeuralMAG speed test #
    #########################
    # NeuralMAG spin calc speed test
    # skip
    # Unet Hd calculation speed test
    if args.profile=='True':
        with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=profiler.tensorboard_trace_handler(f'./log_hd_nccl_{args.w}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as p:
            for i in range(args.n_loop):
                if i==10:
                    for d in args.gpu:
                        torch.cuda.synchronize(d)
                    start_time = time.time()

                MAG2305.MFNN(film2.Spin)
            p.step()

            for d in args.gpu:
                torch.cuda.synchronize(d)
            end_time = time.time()

    else:
        for i in range(args.n_loop):
            if i==10:
                for d in args.gpu:
                    torch.cuda.synchronize(d)
                start_time = time.time()

            MAG2305.MFNN(film2.Spin)

        for d in args.gpu:
            torch.cuda.synchronize(d)
        end_time = time.time()

    Hd_speed = (end_time - start_time) / (args.n_loop-10) * 4
    print('done')
    MAG2305.close()
    if args.trt=='True':
        print(f'||Unt_trt:  {args.w} || Spin calc speed: {Hd_speed:.1e} s || Hd calc speed: {Hd_speed:.1e} s||')
    else:
        print(f'||Unt_halo_size: {args.w} || Spin calc speed: {Hd_speed:.1e} s || Hd calc speed: {Hd_speed:.1e} s||')
