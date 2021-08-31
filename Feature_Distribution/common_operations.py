import os
import h5py
import vast
import torch
import torch.distributed.rpc as rpc
from vast.tools import logger as vastlogger
from vast.opensetAlgos import save_load_operations as save_load_operations
from typing import Generator, List, Dict, Tuple

model_saver_obj = None
class save_inference_results:
    def __init__(self, args,
                 process_combination: Tuple[str, int],
                 total_no_of_classes: int,
                 output_file_name: str = None) -> None:
        self.total_no_of_classes = total_no_of_classes
        self.processed_classes = 0
        self.hf = h5py.File(output_file_name, "w")

    def __call__(self, cls_name, data):
        layer_name, values = data
        g = self.hf.create_group(cls_name)
        g.create_dataset(layer_name, data=values)
        self.processed_classes+=1

    def wait(self):
        while True:
            if self.processed_classes >= self.total_no_of_classes:
                break
        return

    def close(self):
        self.hf.close()

def saver_process_execution(cls_name, data):
    model_saver_obj(cls_name, data)

def saver_process_initialization(rank, args, total_no_of_classes = None, savers_mapping = None, output_file_name=None):
    global model_saver_obj
    world_size = args.world_size + len(savers_mapping)
    if args.world_size>1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port_no
        rpc.init_rpc(f"{rank}",
                     rank=rank,
                     world_size = world_size,
                     backend=rpc.BackendType.TENSORPIPE,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                         init_method='env://')
                     )
        logger = vastlogger.get_logger(level=args.verbose, output=args.output_dir,
                                        distributed_rank=rank, world_size=world_size)
        logger.info(f"Started RPC Saver process")
    else:
        logger = vastlogger.get_logger()
    if output_file_name is None:
        logger.debug(f"Calling model saver")
        saver_class = save_load_operations.model_saver
    else:
        logger.debug(f"Calling inference file saver")
        saver_class = save_inference_results
    model_saver_obj = saver_class(args,
                                  (savers_mapping[rank], rank), #args.param_comb_to_saver_mapping
                                  total_no_of_classes,
                                  output_file_name)
    model_saver_obj.wait()
    model_saver_obj.close()
    logger.info(f"Shutting down RPC saver process for combination {savers_mapping[rank]}")
    rpc.shutdown()
    return

def call_distance_based_approaches(gpu, args, features_all_classes, logger, models,
                                   new_classes_to_add = None):
    if new_classes_to_add is None:
        class_names = list(features_all_classes.keys())
        exemplar_classes = []
        for _ in class_names:
            if 'exemplars_' in _:
                exemplar_classes.append(_)
        if len(exemplar_classes):
            logger.info(" Removing Exemplars from positive classes to be processed ".center(90, '#'))
        class_names = sorted(list(set(class_names)-set(exemplar_classes)))
    else:
        class_names = new_classes_to_add
    div, mod = divmod(len(class_names), args.world_size)
    pos_classes_to_process = class_names[gpu * div + min(gpu, mod):(gpu + 1) * div + min(gpu + 1, mod)]

    # pos_classes_to_process = pos_classes_to_process[:2]
    logger.debug(f"Processing classes {pos_classes_to_process}")

    if models is None:
        OOD_Method = getattr(vast.opensetAlgos, args.OOD_Algo + '_Training')
        logger.info(f"Calling approach {OOD_Method.__name__}")
        algorithm_results_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
        for current_class_output in algorithm_results_iterator:
            param_comb, (cls_name, model) = current_class_output
            model['weibulls'] = model['weibulls'].return_all_parameters()
            if 'extreme_vectors' in model:
                del model['extreme_vectors']

            _ = rpc.remote(f"{args.param_comb_to_saver_mapping_reverse[param_comb]}",
                           saver_process_execution,
                           timeout=0,
                           args=(cls_name, model))
    else:
        OOD_Method = getattr(vast.opensetAlgos, args.OOD_Algo + '_Inference')
        logger.info(f"Calling approach {OOD_Method.__name__}")
        algorithm_results_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
        for current_class_output in algorithm_results_iterator:
            layer_name, (cls_name, probs) = current_class_output
            _ = rpc.remote(f"{args.world_size}",
                           saver_process_execution,
                           timeout=0,
                           args=(cls_name, (layer_name, probs)))
    logger.debug(f"Shutting down")
    rpc.shutdown()
    return

def call_specific_approach(rank, args, features_all_classes,
                           models=None, new_classes_to_add = None):
    if args.world_size>1:
        if models is None:
            world_size = args.world_size + len(args.param_comb_to_saver_mapping)
        else:
            world_size = args.world_size + 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port_no
        rpc.init_rpc(f"{rank}",
                     rank=rank,
                     world_size=world_size,
                     backend=rpc.BackendType.TENSORPIPE,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                         init_method='env://')
                     )
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
        logger = vastlogger.get_logger(level=args.verbose, output=args.output_dir,
                                        distributed_rank=rank, world_size=world_size)
        logger.debug(f"Started process")
    else:
        logger = vastlogger.get_logger()
    return call_distance_based_approaches(rank, args, features_all_classes, logger,
                                          models, new_classes_to_add)