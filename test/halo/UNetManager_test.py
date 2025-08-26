import multiprocessing as mp

import torch
from torch import nn
from UNet_nccl_test import UNet
import torch.profiler as profiler
class UNetManager:
    def __init__(self, kc, inc, ouc, device_list, split, cpkt, input_shape):
        self.model_args = (kc, inc, ouc, device_list, split)

        self.cpkt = cpkt
        self.nproc = len(device_list)
        self.processes = []
        self.task_queues = [mp.Queue() for _ in range(self.nproc)]
        self.result_queue = [mp.Queue() for _ in range(self.nproc)]
        self.input_shm = torch.empty(input_shape, device='cpu').share_memory_()
        self.output_shm = torch.empty(input_shape, device='cpu').share_memory_()
        self._start_workers()


    def _start_workers(self):
        for rank in range(self.nproc):
            p = mp.Process(target=self._worker_fn, 
                           args=(rank, 
                                 self.model_args, 
                                 self.cpkt, 
                                 self.task_queues[rank], 
                                 self.result_queue[rank],
                                 self.input_shm,
                                 self.output_shm))
            p.start()
            self.processes.append(p)

    @staticmethod
    def _worker_fn(rank, model_args, cpkt, task_queues, result_queue, input_shm, output_shm):
        model = UNet(*model_args).eval()
        model.load_weights(cpkt)
        model._setup(rank, len(model_args[3]))
        device = model_args[3][rank]
        split_y, split_x = model_args[4]
        # with profiler.profile(
        #     activities=[
        #         profiler.ProfilerActivity.CPU,
        #         profiler.ProfilerActivity.CUDA,
        #     ],
        #     on_trace_ready=profiler.tensorboard_trace_handler(f'./log_rank_{rank}'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as p:
        while True:
            signal = task_queues.get()
            if signal is None:
                break 

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
            model._unet_dist(rank, local_input, local_output, result_queue)
                # p.step()
    def predict(self, input_tensor: torch.Tensor):
        self.input_shm.copy_(input_tensor)
        for rank, q in enumerate(self.task_queues):
            q.put("start")

        for rank,q in enumerate(self.result_queue):
            q.get()
        return self.output_shm

    def close(self):
        for q in self.task_queues:
            q.put(None)
        for p in self.processes:
            p.join()
