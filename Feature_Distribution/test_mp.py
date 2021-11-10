"""
This code can be used to test if the rpc based code in Feature_Distribution.py will work on your machine.
It was also used for the following question
https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772
"""
import os
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

world_size=torch.cuda.device_count()
# Set this number to the total number of hyper parameter combinations in your case, from Feature_Distribution.py
no_of_saver_processes=50

def cpu_process_initialization(rank):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='9867'
    print(f"cpu rank {rank}")
    rpc.init_rpc(f"{rank}",
                 rank=rank,
                 world_size=world_size+no_of_saver_processes,
                 backend=rpc.BackendType.TENSORPIPE,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                     init_method='env://')
                 )
    pid = os.getpid()
    print(f"Started CPU process {rank} with pid {pid}")
    print(f"Process {rank}: avaialable device {torch.cuda.current_device()}")
    print(f'sudo /proc/{pid}/fd | wc -l')
    stream = os.popen(f'sudo ls /proc/{pid}/fd | wc -l')
    output = stream.read()
    print(f"{rank}/{pid} No of file descriptors in use {output}")

    # Do something rather than sleeping example disk or cpu bound operations
    time.sleep(60)
    rpc.shutdown()
    return

def cuda_process_initialization(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9867'
    rpc.init_rpc(f"{rank}",
                 rank=rank,
                 world_size=world_size + no_of_saver_processes,
                 backend=rpc.BackendType.TENSORPIPE,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                     init_method='env://')
                 )
    pid = os.getpid()
    torch.cuda.set_device(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    print(f"Started CUDA process on gpu {rank} with pid {pid}")
    # Do some cuda operations
    print(f"Process {rank}: avaialable device {torch.cuda.current_device()}")
    stream = os.popen(f'sudo ls /proc/{pid}/fd | wc -l')
    output = stream.read()
    print(f"{rank}/{pid} No of file descriptors in use {output}")
    time.sleep(60)
    rpc.shutdown()
    return

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    trainer_processes = mp.spawn(cuda_process_initialization,
                                 nprocs=world_size,
                                 join=False)
    cpu_processes = []
    for rank in range(world_size,world_size+no_of_saver_processes):
        p = mp.Process(target=cpu_process_initialization,
                       args=(rank,))
        p.start()
        cpu_processes.append(p)
    for p in cpu_processes: p.join()
    trainer_processes.join()
