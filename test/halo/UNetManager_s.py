import multiprocessing as mp
from datetime import datetime
from threading import local
from time import sleep

import torch
from torch import nn
from UNet_nccl_s import UNet
import torch.profiler as profiler
class UNetManager:
    def __init__(self, kc, inc, ouc, device_list, split, cpkt, input_shape):
        self.model_args = (kc, inc, ouc, device_list, split)

        self.cpkt = cpkt
        self.nproc = len(device_list)
        self.processes = []
        self.input_flag = torch.zeros(self.nproc, dtype=torch.int8, device='cpu').share_memory_()
        self.output_flag = torch.zeros(self.nproc, dtype=torch.int8, device='cpu').share_memory_()
        self.input_shm = torch.empty(input_shape, device='cpu').share_memory_()
        self.output_shm = torch.empty(input_shape, device='cpu').share_memory_()
        self._start_workers()


    def _start_workers(self):
        for rank in range(self.nproc):
            p = mp.Process(target=self._worker_fn, 
                           args=(rank, 
                                 self.model_args, 
                                 self.cpkt, 
                                 self.input_flag, 
                                 self.output_flag,
                                 self.input_shm,
                                 self.output_shm))
            p.start()
            self.processes.append(p)

    @staticmethod
    def _worker_fn(rank, model_args, cpkt, input_flag, output_flag, input_shm, output_shm):
        model = UNet(*model_args).eval()
        model.load_weights(cpkt)
        model._setup(rank, len(model_args[3]))
        device = model_args[3][rank]
        split_y, split_x = model_args[4]
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=profiler.tensorboard_trace_handler(f'./log_rank_{rank}'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as p:
            while True:
                # t1 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                # print(rank, "before get input", t1, flush=True)

                while input_flag[rank] == 0:
                    continue

                if input_flag[rank] == -1:
                    break
                
                # t2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                # print(rank, "after get input", t2, flush=True)

                _, _, H, W = input_shm.shape
                i = rank // split_x
                j = rank % split_x
                split_H = H // split_y
                split_W = W // split_x
                local_input = input_shm[:, :, 
                                        i*split_H:(i+1)*split_H, 
                                        j*split_W:(j+1)*split_W].to(device)
                local_output = output_shm[:, :,
                                        i*split_H:(i+1)*split_H, 
                                        j*split_W:(j+1)*split_W]
                model._unet_dist(rank, local_input, local_output, output_flag)
                # t4 = datetime.now().strftime("%H:%M:%S.%f")[:-3] 
                # print(rank, "after model forward", t4, flush=True)
                p.step()

    def predict(self, input_tensor: torch.Tensor):
        # t1 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # print("start predict", t1, flush=True)
        self.input_shm.copy_(input_tensor)
        # t2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # print("after share memory", t2, flush=True)
        self.input_flag.fill_(1)

        # t3 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # print(rank, "before get output", t3)

        while self.output_flag.sum() < self.nproc:
            continue
        self.output_flag.fill_(0)
        # t4 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # print(rank, "got output", t4)

        return self.output_shm

    def close(self):
        self.input_flag.fill_(-1)
        for p in self.processes:
            p.join()