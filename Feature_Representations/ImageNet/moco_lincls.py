# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
import math
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import pathlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from vast import tools
from vast import losses
from torch.utils.tensorboard import SummaryWriter

import csv
import numpy as np
from torchvision.datasets.folder import pil_loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', type=str,
                    default='/scratch/datasets/ImageNet/ILSVRC_2012',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument("--output-path", help="output directory path", default="/net/reddwarf/bigscratch/adhamija/The/FR/",
                    required=False)
parser.add_argument("--approach", default=None, required=True,
                    choices=['SoftMax', 'CenterLoss', 'COOL', 'BG', 'entropic', 'objectosphere', 'RBF_SoftMax',
                             'arcface', 'sphereface', 'cosface'])
parser.add_argument('--second_loss_weight', help='Loss weight for Objectosphere loss', type=float, default=1.0)
parser.add_argument('--Minimum_Knowns_Magnitude', help='Minimum Possible Magnitude for the Knowns', type=float,
                    default=50.)
parser.add_argument('--debug', action='store_true', default=False,
                    help='Run only 2 iterations per gpu')
parser.add_argument('--objecto_layer', help='Objectosphere layer dimension', type=int, default=None)

best_acc1 = 0

global known_dataset, known_unknowns_dataset
global val_known_dataset, val_known_unknowns_dataset

class OpensetDataset(torch.utils.data.Dataset):
    def __init__(self, protocol_file, img_dir, transform=None, debug=False):
        self.data = list(csv.reader(open(protocol_file), delimiter=" "))
        classes = sorted(set([d[0].split('/')[1] for d in self.data]))
        self.classes_mapping = dict(zip(classes,list(range(len(classes)))))
        if debug:
            self.data = self.data[:500]
        self.img_dir = img_dir
        self.transforms = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, _ = self.data[idx]
        label = self.classes_mapping[file_name.split('/')[1]]
        image = pil_loader(f"{self.img_dir}/{file_name}")
        if self.transforms:
            image = self.transforms(image)
        return image, np.array(int(label))


def get_loss_functions(args, known_dataset, val_known_dataset, known_unknowns_dataset, val_known_unknowns_dataset):
    approach = {"SoftMax": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                                second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                dir_name="Softmax",
                                training_data=[known_dataset],
                                val_data=[val_known_dataset]
                                ),
                "CenterLoss": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                                   second_loss_func=losses.tensor_center_loss(beta=0.1, classes=range(320), fc_layer_dimension= 2048 if args.objecto_layer is None else args.objecto_layer),
                                   dir_name="CenterLoss",
                                   training_data=[known_dataset],
                                   val_data=[val_known_dataset]
                                   ),
                "COOL": dict(first_loss_func=losses.entropic_openset_loss(),
                             second_loss_func=losses.objecto_center_loss(
                                 beta=0.1,
                                 classes=range(-1, 320, 1),
                                 ring_size=args.Minimum_Knowns_Magnitude),
                             dir_name=f"COOL",
                             training_data=[known_dataset, known_unknowns_dataset],
                             val_data=[val_known_dataset, val_known_unknowns_dataset]
                             ),
                "BG": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                           second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                           dir_name="BGSoftmax",
                           training_data=[known_dataset, known_unknowns_dataset],
                           val_data=[val_known_dataset, val_known_unknowns_dataset]
                           ),
                "entropic": dict(first_loss_func=losses.entropic_openset_loss(num_of_classes=320),
                                 second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                 dir_name=f"Cross",
                                 training_data=[known_dataset, known_unknowns_dataset],
                                 val_data=[val_known_dataset, val_known_unknowns_dataset]
                                 ),
                "objectosphere": dict(first_loss_func=losses.entropic_openset_loss(num_of_classes=320),
                                      second_loss_func=losses.objectoSphere_loss(
                                          args.batch_size,
                                          knownsMinimumMag=args.Minimum_Knowns_Magnitude),
                                      dir_name=f"ObjectoSphere/{args.Minimum_Knowns_Magnitude}",
                                      training_data=[known_dataset, known_unknowns_dataset],
                                      val_data=[val_known_dataset, val_known_unknowns_dataset]
                                      ),
                "RBF_SoftMax": dict(first_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                    second_loss_func=losses.RBFLogits(feature_dim=2, class_num=10, scale=4.0,
                                                                      gamma=1.0),
                                    dir_name="RBF_SoftMax",
                                    training_data=[known_dataset],
                                    val_data=[val_known_dataset]
                                    ),
                "arcface": dict(first_loss_func=losses.AngularPenaltySMLoss(2, 10, loss_type='arcface', eps=1e-7, s=None, m=None),
                                second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                dir_name="arcface",
                                training_data=[known_dataset],
                                val_data=[val_known_dataset]
                                ),
                "sphereface": dict(first_loss_func=losses.AngularPenaltySMLoss(2, 10, loss_type='sphereface',
                                                                               eps=1e-7, s=None, m=None),
                                   second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                   dir_name="sphereface",
                                   training_data=[known_dataset],
                                   val_data=[val_known_dataset]
                                   ),
                "cosface": dict(first_loss_func=losses.AngularPenaltySMLoss(2, 10, loss_type='cosface',
                                                                            eps=1e-7, s=None, m=None),
                                second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                dir_name="cosface",
                                training_data=[known_dataset],
                                val_data=[val_known_dataset]
                                )
                }
    return approach[args.approach]

def main():
    args = parser.parse_args()
    print('\n\n')
    print('#' * 180)
    print(f"\n{' '.join(sys.argv)}\n")
    print('#' * 180)
    print('\n\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


class Model_Operations():
    def __init__(self, args, objecto_layer=None, bias_flag=True, no_of_classes=320):
        model = models.__dict__[args.arch]()

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        if args.approach == "BG":
            no_of_classes+=1
        if objecto_layer is not None:
            model.fc = nn.Sequential(nn.Linear(model.fc.in_features, objecto_layer),
                                     nn.Linear(objecto_layer, no_of_classes, bias=bias_flag))
            # init the fc layer
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
            model.fc[1].weight.data.normal_(mean=0.0, std=0.01)
            if bias_flag:
                model.fc[1].bias.data.zero_()
        else:
            model.fc = nn.Linear(model.fc.in_features, no_of_classes)

            # init the fc layer
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

        self.model = model
        self.objecto_layer = objecto_layer
        self.register_hooks(args)
        print(f"\n\n\n{self.model}\n\n\n")

    def feature_hook(self, module, input, output):
        self.outputs.append(output)

    def register_hooks(self, args):
        if self.objecto_layer is not None:
            self.model.fc[0].register_forward_hook(self.feature_hook)
        else:
            self.model.avgpool.register_forward_hook(self.feature_hook)
        self.model.fc.register_forward_hook(self.feature_hook)

    def __call__(self, x):
        self.outputs = []
        _ = self.model(x)
        features, Logit = self.outputs
        features = features.view(-1, features.shape[1])
        return Logit, features


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    known_dataset = OpensetDataset(protocol_file='../protocol/train_knowns.csv',
                                   img_dir=args.data,
                                   transform=transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize
                                   ]),
                                   debug=args.debug)
    known_unknowns_dataset = OpensetDataset(protocol_file='../protocol/train_knownUnknowns.csv',
                                            img_dir=args.data,
                                            transform=transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize
                                            ]),
                                            debug=args.debug)
    val_known_dataset = OpensetDataset(protocol_file='../protocol/test_knowns.csv',
                                       img_dir=args.data,
                                       transform=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]),
                                       debug=args.debug)
    val_known_unknowns_dataset = OpensetDataset(protocol_file='../protocol/test_knownUnknowns.csv',
                                                img_dir=args.data,
                                                transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                ]),
                                                debug=args.debug)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))

    modelObj = Model_Operations(args,
                                objecto_layer=args.objecto_layer,
                                bias_flag=True,
                                no_of_classes=320)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = modelObj.model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            modelObj.model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            modelObj.model = torch.nn.parallel.DistributedDataParallel(modelObj.model, device_ids=[args.gpu])
        else:
            modelObj.model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            modelObj.model = torch.nn.parallel.DistributedDataParallel(modelObj.model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        modelObj.model = modelObj.model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            modelObj.model.features = torch.nn.DataParallel(modelObj.model.features)
            modelObj.model.cuda()
        else:
            modelObj.model = torch.nn.DataParallel(modelObj.model).cuda()

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, modelObj.model.parameters()))
    print(f"parameters being trained\n{[p.shape for p in parameters]}")
    # assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            modelObj.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    first_loss_func,second_loss_func,dir_name,training_data,validation_data = \
        list(zip(*get_loss_functions(args, known_dataset, val_known_dataset,
                                     known_unknowns_dataset, val_known_unknowns_dataset).items()))[-1]

    if args.debug:
        args.output_path="/tmp/"
    parent_results_dir = f"{args.output_path}/{args.arch}/{args.objecto_layer}/{dir_name}/"

    if args.cos:
        results_dir = pathlib.Path(f"{parent_results_dir}/cos_{args.lr}_{args.second_loss_weight}/")
    else:
        results_dir = pathlib.Path(f"{parent_results_dir}/{args.lr}_{args.second_loss_weight}/")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Data loading code
    train_dataset = tools.ConcatDataset(training_data,BG=args.approach == "BG", no_of_classes=320)
    val_dataset = tools.ConcatDataset(validation_data,BG=args.approach == "BG", no_of_classes=320)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    # print(f"val_loader {len(val_loader)}")
    # print(f"train_loader {len(train_loader)}")

    if args.evaluate:
        validate(val_loader, modelObj, criterion, args)
        return

    saver_process = False
    tensorboard_writer = None
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        saver_process = True
        tensorboard_writer = SummaryWriter(results_dir / 'logs')

    if saver_process:
        torch.save(modelObj.model.state_dict(), f"{parent_results_dir}/architecture.pth.tar")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        _=common_processing(args, train_loader, modelObj, first_loss_func, second_loss_func,
                            optimizer, epoch,
                            tensorboard_writer=tensorboard_writer)

        # evaluate on validation set
        with torch.no_grad():
            acc1 = common_processing(args, val_loader, modelObj, first_loss_func, second_loss_func,
                                     optimizer=None, epoch_no=epoch,
                                     tensorboard_writer=tensorboard_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if saver_process:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': modelObj.model,#.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, dir_name=results_dir)
            # if epoch == args.start_epoch:
            #     sanity_check(modelObj.model.state_dict(), args.pretrained)


def common_processing(args, data_loader, modelObj, first_loss_func, second_loss_func, optimizer=None, epoch_no=None,
                      tensorboard_writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    avg_meter_first_loss = AverageMeter('FirstLoss', ':.4e')
    avg_meter_second_loss = AverageMeter('SecondLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(data_loader),
                             [avg_meter_first_loss, avg_meter_second_loss, top1, top5],
                             prefix="Test: " if optimizer is None else "Epoch: [{}]".format(epoch_no))
    # progress = ProgressMeter(len(data_loader),
    #                          [batch_time, data_time, avg_meter_first_loss, avg_meter_second_loss, top1, top5],
    #                          prefix="Test: " if optimizer is None else "Epoch: [{}]".format(epoch_no))
    modelObj.model.eval()
    end = time.time()
    vast_accuracy = torch.zeros(2, dtype=int)
    vast_magnitude = torch.zeros(2, dtype=float)
    vast_confidence = torch.zeros(2, dtype=float)

    epoch_running_loss = []
    for i, (images, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        logits, features = modelObj(images)
        first_loss = first_loss_func(logits, target)
        second_loss = torch.tensor(0.)
        if optimizer is not None and epoch_no < 5 and args.approach in ["COOL", "CenterLoss"]:
            loss = first_loss
        else:
            second_loss = second_loss_func(features, target)
            loss = first_loss + args.second_loss_weight * second_loss

        # VAST specific metrics
        vast_accuracy += losses.accuracy(logits, target)
        vast_confidence += losses.confidence(logits, target)
        if args.approach not in ("SoftMax", "BG"):
            vast_magnitude += losses.sphere(features, target,
                                            args.Minimum_Knowns_Magnitude if args.approach in ("COOL", "Objectosphere") else None)
        epoch_running_loss.extend(loss.tolist())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        avg_meter_first_loss.update(first_loss.mean().item(), images.size(0))
        avg_meter_second_loss.update(second_loss.mean().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if optimizer is not None:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        if args.approach in ["CenterLoss", "COOL"]:
            second_loss_func.update_centers(features, target)
            modelObj.model.register_parameter('centers',
                                              torch.nn.Parameter(second_loss_func.centers.clone(), requires_grad=False))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    epoch_running_loss=torch.mean(torch.tensor(epoch_running_loss)).item()

    train_or_val = 'train' if optimizer is not None else 'val'
    to_record = {}
    for meter in progress.meters:
        if type(meter.__dict__['avg'])==float:
            to_record[meter.name] = meter.avg
        else:
            to_record[meter.name] = meter.avg.item()
    to_record['loss'] = epoch_running_loss
    to_record['vast_accuracy'] = float(vast_accuracy[0]) / float(vast_accuracy[1])
    to_record['vast_confidence'] = vast_confidence[0] / vast_confidence[1]
    if vast_magnitude[1]:
        to_record['magnitude'] = vast_magnitude[0] / vast_magnitude[1]

    if tensorboard_writer is not None:
        for r in to_record:
            tensorboard_writer.add_scalar(f"{train_or_val}/{r}", to_record[r], global_step=epoch_no)
        if optimizer is not None:
            tensorboard_writer.add_scalar(f"{train_or_val}/lr", list(optimizer.param_groups)[0]['lr'], global_step=epoch_no)

    print(f"Vast metrics "
          f"loss {epoch_running_loss:.10f} "
          f"accuracy {float(vast_accuracy[0]) / float(vast_accuracy[1]):.5f} "
          f"confidence {vast_confidence[0] / vast_confidence[1]:.5f} "
          f"magnitude {vast_magnitude[0] / vast_magnitude[1] if vast_magnitude[1] else -1:.5f} -- "
          )
    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if torch.isnan(torch.tensor(epoch_running_loss)).item():
        raise Exception("Exiting ... loss became nan")
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dir_name='.'):
    learnable_params_to_save = {}
    original_state_dict = state['state_dict'].state_dict()
    for name, param in state['state_dict'].named_parameters():
        if param.requires_grad:
            learnable_params_to_save[name] = original_state_dict[name]
    state['state_dict'] = learnable_params_to_save
    torch.save(state, f"{dir_name}/{filename}")
    if is_best:
        shutil.copyfile(f"{dir_name}/{filename}", f'{dir_name}/model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
