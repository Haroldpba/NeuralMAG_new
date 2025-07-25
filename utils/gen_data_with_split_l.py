# -*- coding: utf-8 -*-
import random
import time, os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import matplotlib.pyplot as plt

from libs.misc import Culist, initial_spin_prepare, create_random_mask, error_plot, wd_plot
import libs.MAG2305 as MAG2305
def plot_spin(spin, save_path):
    plt.cla()
    spin_plt = (np.array(spin) + 1)/2  # spin range (-1,1)
    plt.imshow( spin_plt.transpose(1,0,2), origin='lower' )
    plt.savefig(save_path+'/spin.png')
def winding_density(spin_batch):
    """
    用于计算batch数据的winding density
    Args:
    spin_batch: torch.tensor
                形状为(batch_size, 3, 32, 32)的tensor，表示包含batch_size个样本的spin数据
    Returns:
    winding_density_batch: torch.tensor
                形状为(batch_size, 32, 32)的tensor，表示batch数据的winding density
    """
    # 调整spin的维度顺序为[batch_size, 32, 32, 1, 3]
    spin = spin_batch.clone().detach().permute(0, 2, 3, 1).unsqueeze(-2)
    spin_xp = torch.roll(spin, shifts=-1, dims=1)
    spin_xm = torch.roll(spin, shifts=1, dims=1)
    spin_yp = torch.roll(spin, shifts=-1, dims=2)
    spin_ym = torch.roll(spin, shifts=1, dims=2)
    spin_xp[:, -1, :, :, :] = spin[:, -1, :, :, :]
    spin_xm[:, 0, :, :, :]  = spin[:, 0, :, :, :]
    spin_yp[:, :, -1, :, :] = spin[:, :, -1, :, :]
    spin_ym[:, :, 0, :, :]  = spin[:, :, 0, :, :]
    winding_density = (spin_xp[:,:,:, 0, 0] - spin_xm[:,:,:, 0, 0]) / 2 * (spin_yp[:,:,:, 0, 1] - spin_ym[:,:,:, 0, 1]) / 2 \
                    - (spin_xp[:,:,:, 0, 1] - spin_xm[:,:,:, 0, 1]) / 2 * (spin_yp[:,:,:, 0, 0] - spin_ym[:,:,:, 0, 0]) / 2
    
    winding_density = winding_density / np.pi
    winding_abs = torch.abs(winding_density).sum(dim=(1,2))

    return winding_density, torch.round(winding_abs).cpu().numpy()

def prepare_model(args):
    film = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), 
                           cell=(3,3,3), Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, 
                           Kvec=args.Kvec, device="cuda:" + str(args.gpu))
    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    film.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for initializing demag matrix \n'.format(time_finish-time_start))
    return film

def generate_data(args, film):
    Hext_val = np.random.randn(3) * args.Hext_val
    Hext = Hext_val * args.Hext_vec

    for seed in tqdm(range(args.nstart, args.nseeds+args.nstart)):
        path_format = './Dataset/data_Hd{}_Hext{}_mask/seed{}_split{}' if args.mask=='True' else './Dataset_d/data_Hd{}_Hext{}/seed{}_split{}'
        save_path = path_format.format(args.w, int(args.Hext_val), seed, args.split)
        os.makedirs(save_path, exist_ok=True)
        spin = MAG2305.get_randspin_2D(size=(args.w, args.w, args.layers),
                                split=args.split, rand_seed=seed)
        if seed == 0:
            print(spin.shape)
            plot_spin(spin[:,:,0,:], save_path)
        if args.mask == 'True':
            mask = create_random_mask((args.w, args.w), np.random.randint(2, args.w), random.choice([True, False]))
            spin = film.SpinInit(spin * mask)
        else:
            spin = film.SpinInit(spin)

        error_list, wd_list = simulate_spins(film, spin, Hext, args, save_path)
        save_simulation_data(args, save_path, error_list, wd_list, Hext)

def simulate_spins(film, spin, Hext, args, save_path):
    error_list = []
    wd_list = []
    itern = 1
    error_ini = 1

    Spininit = np.reshape(spin[:,:,:,:], (args.w, args.w, args.layers*3))
    np.save(os.path.join(save_path, f'Spins_0.npy'), Spininit)
    pbar = tqdm(total=args.max_iter)
    while error_ini > args.error_min and itern < 1001:
        error = film.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)
        error_ini = error
        error_list.append(error)
        _, wd = winding_density(np.reshape(film.Spin.cpu(), (1, args.w, args.w, args.layers*3)).permute(0, 3, 1, 2))
        wd_list.append(wd[0])
        np.save(os.path.join(save_path, f'Spins_{itern}.npy'), np.reshape(film.Spin.cpu(), (args.w, args.w, args.layers*3)))
        np.save(os.path.join(save_path, f'Hds_{itern}.npy'), np.reshape(film.Hd.cpu(), (args.w, args.w, args.layers*3)))
        itern += 1
        pbar.update(1)
        pbar.set_description(f"error: {error}")
    pbar.close()
    return error_list, wd_list

def save_simulation_data(args, save_path, error_list, wd_list, Hext):
    # 获取所有保存的文件
    spin_files = sorted([f for f in os.listdir(save_path) if f.startswith('Spins_') and f.endswith('.npy')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    hd_files = sorted([f for f in os.listdir(save_path) if f.startswith('Hds_') and f.endswith('.npy')], key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 随机选取所需的样本
    random_indices = random.sample(range(len(hd_files)-1), 500)
    Spins_random_list = [np.load(os.path.join(save_path, spin_files[i])) for i in random_indices]
    Hds_random_list = [np.load(os.path.join(save_path, hd_files[i])) for i in random_indices]

    # 删除所有的文件
    for f in spin_files:
        os.remove(os.path.join(save_path, f))
    for f in hd_files:
        os.remove(os.path.join(save_path, f))
    # 保存数据
    np.save(os.path.join(save_path, 'Spins.npy'), np.stack(Spins_random_list, axis=0))
    np.save(os.path.join(save_path, 'Hds.npy'), np.stack(Hds_random_list, axis=0))
    # 保存error和wd数据到txt
    np.savetxt(os.path.join(save_path, 'wd.txt'), wd_list)

    error_plot(error_list, os.path.join(save_path, 'iterns{:.1e}_errors_{:.1e}'.format(len(error_list), error_list[-1])),
               str('[{:.2f}, {:.2f}, {:.2f}]'.format(Hext[0], Hext[1], Hext[2])))
    wd_plot(wd_list, os.path.join(save_path, 'iterns{:.1e}_winding_density_{:.1e}'.format(len(wd_list), wd_list[-1])),
            str('[{:.2f}, {:.2f}, {:.2f}]'.format(Hext[0], Hext[1], Hext[2])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')
    parser.add_argument('--split',     type=int,    default=4,         help='MAG model split (default: 4)')
    
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-6,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples',type=int,    default=500,       help='save samples (default: 500)')
    parser.add_argument('--mask',       type=str,    default='False',   help='mask (default: False)')
    parser.add_argument('--nseeds',     type=int,    default=100,       help='number of seeds (default: 100)')
    parser.add_argument('--nstart',     type=int,    default=0,         help='seed (default: 0)')

    args = parser.parse_args() 

    device = torch.device("cuda:{}".format(args.gpu))
    
    #Prepare MAG model: film
    film = prepare_model(args)

    #Generate spin and Hd pairs data
    generate_data(args, film)
