"""
Runs ODIN with hyper parameter grid search
Currently only supports ImageNet completely but can do MNIST after small modifications
"""
import os
import argparse
import numpy as np
import pathlib
import itertools
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import common_operations
from vast import opensetAlgos
from vast.tools import logger as vastlogger
from vast.data_prep import readHDF5
from vast import architectures

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     add_help=False, usage=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', help="To decrease verbosity increase", action='count', default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="debugging flag")
    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing\ndefault: %(default)s')
    parser.add_argument("--output_dir", type=str, default='/tmp/adhamija/', help="Results directory")
    parser.add_argument('--OOD_Algo', default='ODIN', type=str, choices=['ODIN'],
                        help='Name of the OOD detection algorithm')
    parser.add_argument("--arch",
                        default='LeNet_plus_plus',
                        help="The architecture from which to extract layers. "
                             "Can be a model architecture already available in torchvision or a saved pytorch model.")
    parser.add_argument("--images-path", help="directory containing imagenet images",
                        default="/scratch/datasets/ImageNet/", required=False)
    parser.add_argument("--weights", help="network weights",
                        default=None, required=False)


    known_args, unknown_args = parser.parse_known_args()

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs experiments for ODIN")
    parser, algo_params = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    args = parser.parse_args()
    return args, algo_params

def call_approach(gpu, args):
    if args.world_size>1:
        world_size = args.world_size + len(args.param_comb_to_saver_mapping)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port_no
        rpc.init_rpc(f"{gpu}",
                     rank=gpu,
                     world_size=world_size,
                     backend=rpc.BackendType.TENSORPIPE,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                         init_method='env://')
                     )
        torch.cuda.set_device(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        logger = vastlogger.get_logger(level=args.verbose, output=args.output_dir,
                                        distributed_rank=gpu, world_size=world_size)
        logger.debug(f"Started process")
    else:
        logger = vastlogger.get_logger()
    if args.arch=='LeNet_plus_plus':
        net = architectures.LeNet_plus_plus(use_BG=False,num_classes=5,use_bias_for_fc =True)
        dataset = torchvision.datasets.MNIST(root='/tmp/adhamija', train=False,
                                             download=True, transform=transforms.ToTensor())
        args.weights = "/home/adhamija/The/MNIST/LeNet_plus_plus/Softmax/Softmax.model"
    if args.arch != 'LeNet_plus_plus' and args.weights is None:
        net = models.__dict__[args.arch](pretrained=True)
        dataset = torchvision.datasets.ImageFolder(args.images_path,
                                                   transforms.Compose([transforms.Resize(256),
                                                                       transforms.CenterCrop(224),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])
                                                                       ]))
    # else:
    # CSV loader
    algorithm_results_iterator = opensetAlgos.ODIN_Training(dataset, net, args,
                                                            gpu, state_dict=args.weights)
    for current_class_output in algorithm_results_iterator:
        param_comb, (cls_name, probs) = current_class_output
        _ = rpc.remote(f"{args.param_comb_to_saver_mapping_reverse[param_comb]}",
                       common_operations.saver_process_execution,
                       timeout=0,
                       args=(cls_name, ("probs", probs)))
    rpc.shutdown()

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args, algo_params = command_line_options()
    args.output_dir = pathlib.Path(f"{args.output_dir}/{args.OOD_Algo}/")
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)

    args.world_size = torch.cuda.device_count()
    if args.debug:
        args.verbose = 0

    # mnist_trainset = torchvision.datasets.MNIST(root='/tmp/adhamija', train=True,
    #                                            download=True, transform=transforms.ToTensor())
    # mnist_testset = torchvision.datasets.MNIST(root='/tmp/adhamija', train=False,
    #                                            download=True, transform=transforms.ToTensor())

    all_string_comb = [algo_params['param_id_string'].format(*k) for k in itertools.product(args.temperature, args.epsilon)]
    args.param_comb_to_saver_mapping = dict(zip(range(args.world_size, len(all_string_comb) + args.world_size),
                                                all_string_comb))
    ranks, combs = zip(*args.param_comb_to_saver_mapping.items())
    args.param_comb_to_saver_mapping_reverse = dict(zip(combs, ranks))

    logger.info("Starting Training Processes")
    if args.world_size==1:
        # Only for debugging
        iterator = call_approach(0, args)
        for _ in iterator: continue
    else:
        trainer_processes = mp.spawn(call_approach,
                                     args=(args, ),
                                     nprocs=args.world_size,
                                     join=False)

    logger.info("Starting Model Saver Processes")
    saver_processes = []
    for i,rank in enumerate(range(args.world_size,args.world_size+len(args.param_comb_to_saver_mapping))):
        output_file_name = args.output_dir / pathlib.Path(f"{list(args.param_comb_to_saver_mapping_reverse.keys())[i]}/")
        output_file_name.mkdir(parents=True, exist_ok=True)
        output_file_name = output_file_name / pathlib.Path(f"mnist.hdf5")
        p = mp.Process(target=common_operations.saver_process_initialization,
                       args=(rank, args, 1000, args.param_comb_to_saver_mapping,output_file_name))
        p.start()
        saver_processes.append(p)

    logger.info("All processes started ... waiting for their completion")

    trainer_processes.join()
    for p in saver_processes: p.join()
    logger.info("Training Ended")