import argparse
import numpy as np
import pathlib
import itertools
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import common_operations
from vast import opensetAlgos
from vast.tools import logger as vastlogger
from vast.data_prep import readHDF5

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     add_help=False, usage=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', help="To decrease verbosity increase", action='count', default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="debugging flag")
    parser.add_argument("--run_only_test", action="store_true", default=False, help="run_only_test")
    parser.add_argument("--do_not_filter_corrects", action="store_true", default=False,
                        help="By default we consider only correct samples for training, "
                             "if you need to consider all samples use this flag.")
    parser.add_argument("--no_multiprocessing", action="store_true", default=False,
                        help="Use for debugging or running on single GPU")
    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing\ndefault: %(default)s')
    parser.add_argument("--output_dir", type=str, default='/scratch/adhamija/results/', help="Results directory")
    parser.add_argument('--OOD_Algo', default='EVM', type=str, choices=['OpenMax','EVM','Turbo_EVM','MultiModalOpenMax'],
                        help='Name of the openset detection algorithm')

    parser.add_argument("--training_knowns_files", nargs="+", help="HDF5 file path for known images",
                        default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"])
    parser.add_argument("--training_unknowns_files", nargs="+", help="HDF5 file path for unknown images",
                        default=None)
    parser.add_argument("--testing_files", nargs="+", help="HDF5 file path for known images",
                        default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"])
    parser.add_argument("--layer_names", nargs="+", help="The layers to extract from each file", default=["avgpool"])

    known_args, unknown_args = parser.parse_known_args()

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs experiments for incremental learning " 
                                                          "i.e. Table 1 and 2 from the Paper")
    parser, algo_params = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    args = parser.parse_args()
    return args, algo_params

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args, algo_params = command_line_options()
    args.output_dir = pathlib.Path(f"{args.output_dir}/{'_'.join(args.layer_names)}/{args.OOD_Algo}/")
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)

    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True
    if args.debug:
        args.verbose = 0

    comb = []
    for a in algo_params['param_names']:
        if type(args.__dict__[a]) == list:
            comb.append(args.__dict__[a])
        else:
            comb.append([args.__dict__[a]])
    all_string_comb = [algo_params['param_id_string'].format(*k) for k in itertools.product(*comb)]

    args.param_comb_to_saver_mapping = dict(zip(range(args.world_size, len(all_string_comb) + args.world_size),
                                                all_string_comb))
    ranks, combs = zip(*args.param_comb_to_saver_mapping.items())
    args.param_comb_to_saver_mapping_reverse = dict(zip(combs, ranks))

    logger.info("Reading Training Files")
    args.feature_files = args.training_knowns_files
    training_data = readHDF5.prep_all_features_parallel(args)
    training_data = dict([(_, training_data[_]['features']) for _ in training_data])

    if not args.run_only_test:

        if not args.do_not_filter_corrects:
            logger.info("Filtering samples")
            if args.layer_names[0]!='fc':
                logger.info("Reading files ")
                original_layer_to_extract = args.layer_names
                args.feature_files = [_.replace(args.layer_names[0], 'fc') for _ in args.training_knowns_files]
                args.layer_names = ['fc']
                classifier_data = readHDF5.prep_all_features_parallel(args)
                classifier_data = dict([(_, classifier_data[_]['features']) for _ in classifier_data])
                args.layer_names = original_layer_to_extract
            else:
                classifier_data = training_data
            for cls_no, cls_name in enumerate(sorted(classifier_data.keys())):
                filter_result = f"{classifier_data[cls_name].shape} "
                training_data[cls_name]=training_data[cls_name][torch.max(classifier_data[cls_name], dim=1).indices == cls_no,:]
                filter_result+=f"{training_data[cls_name].shape}"
                logger.debug(f"{cls_no}, {cls_name}, {filter_result}")

        logger.info("Starting Training Processes")
        trainer_processes = mp.spawn(common_operations.call_specific_approach,
                                     args=(args, training_data),
                                     nprocs=args.world_size,
                                     join=False)

        logger.info("Starting Model Saver Processes")
        saver_processes = []
        for rank in range(args.world_size,args.world_size+len(args.param_comb_to_saver_mapping)):
            p = mp.Process(target=common_operations.saver_process_initialization,
                           args=(rank, args, len(training_data), args.param_comb_to_saver_mapping, None))
            p.start()
            saver_processes.append(p)

        logger.info("All processes started ... waiting for their completion")

        for p in saver_processes: p.join()
        trainer_processes.join()
        logger.info("Training Ended")

    # Use for testing
    # Load the test feature files
    logger.info("Reading Testing Files")
    all_testing_data=[]
    for testing_file in args.testing_files:
        args.feature_files = [testing_file]
        testing_data = readHDF5.prep_all_features_parallel(args)
        testing_data = dict([(_, testing_data[_]['features']) for _ in testing_data])
        all_testing_data.append([testing_file, testing_data])

    if args.OOD_Algo not in ['EVM','Turbo_EVM']: training_data=None
    # Load the model
    for grid_serach_param_combination in args.param_comb_to_saver_mapping.values():
        logger.info(f"Running Testing for combination {grid_serach_param_combination}")
        OOD_model_dict =  opensetAlgos.save_load_operations.model_loader(args,
                                                                         grid_serach_param_combination,
                                                                         training_data)
        if len(OOD_model_dict.keys())==0:
            logger.critical("Model not trained properly ... Skipping")
            continue
        for testing_file, testing_data in all_testing_data:
            logger.info(f"Running Testing for file {testing_file}")
            output_file_name = args.output_dir/pathlib.Path(f"{grid_serach_param_combination}/"
                                                            f"{testing_file.split('/')[-2]}")
            output_file_name.mkdir(parents=True, exist_ok=True)
            output_file_name = output_file_name/pathlib.Path(f"{testing_file.split('/')[-1]}")
            testing_processes = mp.spawn(common_operations.call_specific_approach,
                                         args=(args, testing_data, OOD_model_dict),
                                         nprocs=args.world_size,
                                         join=False)
            saver_process = mp.Process(target=common_operations.saver_process_initialization,
                                       args=(args.world_size, args, len(testing_data),
                                             {args.world_size:testing_file},
                                             output_file_name))
            saver_process.start()
            saver_process.join()
            testing_processes.join()

