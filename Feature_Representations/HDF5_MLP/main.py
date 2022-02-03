import argparse
import numpy as np
import random
import h5py
import pathlib
import itertools
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
from vast.tools import logger as vastlogger
from vast.data_prep import readHDF5
import network_operations

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-v', '--verbose', help="To decrease verbosity increase", action='count', default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="debugging flag")
    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing\ndefault: %(default)s')
    parser.add_argument('--percentage_samples', default=100.0, type=float,
                        help='percentage of samples to use for training')
    parser.add_argument("--output_dir", type=str, default='/scratch/adhamija/results/', help="Results directory")
    parser.add_argument("--approach", default='SoftMax',
                        choices=['SoftMax', 'CenterLoss', 'COOL', 'BG', 'entropic', 'objectosphere'])
    parser.add_argument('--second_loss_weight', help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument('--fc2_dim', help='Size of the second last hidden layer', type=int, default=None)

    parser.add_argument("--training_knowns_files", nargs="+", help="HDF5 file path for known images",
                        default=["/scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5"])
    parser.add_argument("--training_unknowns_files", nargs="+", help="HDF5 file path for unknown images",
                        default=None)
    parser.add_argument("--testing_files", nargs="+", help="HDF5 file path for known images",
                        default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"])
    parser.add_argument("--layer_names", nargs="+", help="The layers to extract from each file", default=["avgpool"])

    MLP_params_parser = parser.add_argument_group(title="MLP", description="MLP params")
    MLP_params_parser.add_argument("--lr", type=float, default=1e-1,
                                   help="Learning rate to use")
    MLP_params_parser.add_argument("--epochs", type=int, default=50,
                                   help="Number of epochs to train")
    MLP_params_parser.add_argument("--Batch_Size", type=int, default=256,
                                   help="Batch size to use")
    args = parser.parse_args()
    return args


def get_percentage_of_data(data, percentage):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    data_to_return={}
    for cls_name in data:
        ind_of_interest = torch.randint(data[cls_name].shape[0],
                                        (min(int(data[cls_name].shape[0]*(percentage/100)),
                                             data[cls_name].shape[0]),
                                         1))
        data_to_return[cls_name] = data[cls_name].gather(0, ind_of_interest.expand(-1, data[cls_name].shape[1]))
    return data_to_return

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.output_dir = pathlib.Path(f"{args.output_dir}/{'_'.join(args.layer_names)}/"
                                   f"{args.percentage_samples}/{args.approach}/"
                                   f"{args.fc2_dim}_{args.lr}_{args.epochs}_{args.second_loss_weight}")
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)

    args.world_size = torch.cuda.device_count()

    args.feature_files = args.training_knowns_files
    training_data_knowns = readHDF5.prep_all_features_parallel(args)
    training_data_knowns = dict([(_, training_data_knowns[_]['features']) for _ in training_data_knowns])

    if args.training_unknowns_files is not None:
        args.feature_files = args.training_unknowns_files
        training_data_unknowns = readHDF5.prep_all_features_parallel(args)
        training_data_unknowns = dict([(_, training_data_unknowns[_]['features']) for _ in training_data_unknowns])

        with h5py.File(args.training_unknowns_files[0].replace('360_openset','Openset_166-2010'), 'r') as hf:
            classes_in_166 = [*hf]

        known_unknown_classes = list(set([*training_data_unknowns]) - set(classes_in_166))
        for k in list(training_data_unknowns.keys()):
            if k not in known_unknown_classes:
                del training_data_unknowns[k]

        training_data_unknowns = readHDF5.prep_all_features_parallel(args)
        training_data_unknowns = dict([(_, training_data_unknowns[_]['features']) for _ in training_data_unknowns])
        training_data_unknowns = get_percentage_of_data(training_data_unknowns, args.percentage_samples)
    else:
        training_data_unknowns = {}

    training_data_knowns = get_percentage_of_data(training_data_knowns, args.percentage_samples)


    net_obj = network_operations.network(num_classes=1000,
                                         input_feature_size=training_data_knowns[[*training_data_knowns][0]].shape[1],
                                         output_dir=args.output_dir,
                                         fc2_dim=args.fc2_dim)

    net_obj.training(training_data=training_data_knowns,
                     known_unknown_training_data=training_data_unknowns,
                     epochs=args.epochs, lr=args.lr, batch_size=args.Batch_Size, args=args)

    for testing_file in args.testing_files:
        args.feature_files = [testing_file]
        testing_data = readHDF5.prep_all_features_parallel(args)
        testing_data = dict([(_, testing_data[_]['features']) for _ in testing_data])
        prob_dict = net_obj.inference(validation_data=testing_data)
        (args.output_dir / f"{testing_file.split('/')[-2]}").mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output_dir/f"{'/'.join(testing_file.split('/')[-2:])}", "w") as hf:
            for cls_name in prob_dict:
                g = hf.create_group(cls_name)
                g.create_dataset("probs", data=prob_dict[cls_name])
        print(prob_dict[cls_name])
