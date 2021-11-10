"""
Script to create evaluation plots along with support for heuristics
"""
import argparse
import h5py
import pathlib
import torch
from matplotlib import pyplot as plt
from vast.eval import eval
from vast.opensetAlgos import heuristic
from vast.data_prep import readHDF5
from vast.tools import viz
from vast.tools import logger as vastlogger
import numpy as np

colors = viz.colors_global

logger = vastlogger.setup_logger(level=2)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Script tor create evaluation plots along with support for heuristics")
parser.add_argument('-v', '--verbose', help="To decrease verbosity increase", action='count', default=0)
parser.add_argument("--debug", action="store_true", default=False, help="debugging flag\ndefault: %(default)s")
parser.add_argument("--knowns_files", nargs="+", help="HDF5 file path for known images",
                              default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"])
parser.add_argument("--unknowns_files", nargs="+", help="HDF5 file path for unknown images",
                              default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"])
parser.add_argument("--label_names", nargs="+", help="Labels",
                              default=["SoftMax"])
parser.add_argument("--all_layer_names", nargs="+", help="The layers to extract from each file", default=["avgpool"])
parser.add_argument("--output_dir", help="Output directory to save plots in",
                    default=pathlib.Path("Resutls"), type=pathlib.Path)
parser.add_argument("--use-softmax-normalization", nargs="+", default=[False], type=str2bool,
                    help="Perform softmax normalization on values read from HDF5 file")
parser.add_argument('--heuristic_to_execute', nargs='+', type=str, default=None)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=False)

# Beta for F Score
ß = 1

def get_gt_prob(file_name, perform_softmax=False, topk=1, layer_name="avgpool"):
    args.layer_names = [layer_name]
    args.feature_files = [file_name]
    data = readHDF5.prep_all_features_parallel(args)
    all_logits = []
    all_gt = []
    for cls_no, cls_name in enumerate(sorted([*data])):
        logit_or_prob = data[cls_name]["features"][...]
        all_gt.extend([cls_no] * logit_or_prob.shape[0])
        all_logits.extend(logit_or_prob.tolist())
    all_gt = torch.tensor(all_gt)
    all_logits = torch.tensor(all_logits)
    if perform_softmax:
        logger.critical("Performing Softmax normalization")
        softmax_op = torch.nn.Softmax(dim=1)
        all_logits = softmax_op(all_logits)
    # The following is a special case for background class, it is hardcoded for 320 known classes
    if all_logits.shape[1]==321:
        all_logits = all_logits[:,:-1]
    all_predicted = torch.arange(all_logits.shape[1])
    all_predicted = all_predicted.repeat((all_logits.shape[0],1))
    if topk is not None:
        all_logits, all_predicted = torch.topk(all_logits, topk, dim=1)
    return all_gt, all_logits, all_predicted

fig_OSRC, ax_OSRC = plt.subplots()
fig_F1, ax_F1 = plt.subplots()
fig_PR, ax_PR = plt.subplots()
fig_coverage, ax_coverage = plt.subplots()
figlegend = plt.figure()
plot_count=0
for plot_no, (known_file, unknown_file, use_softmax_normalization, layer_name) in enumerate(zip(args.knowns_files,
                                                                                                args.unknowns_files,
                                                                                                args.use_softmax_normalization,
                                                                                                args.all_layer_names)):
    knowns_all_gt, knowns_all_prob, knowns_all_predicted = get_gt_prob(known_file,
                                                                       perform_softmax=use_softmax_normalization,
                                                                       topk=None,
                                                                       layer_name=layer_name)
    unknowns_all_gt, unknowns_all_prob, unknowns_all_predicted = get_gt_prob(unknown_file,
                                                                             perform_softmax=use_softmax_normalization,
                                                                             topk=None,
                                                                             layer_name=layer_name)
    unknowns_all_gt[:]=-1

    all_gt = torch.cat((knowns_all_gt, unknowns_all_gt))
    all_prob = torch.cat((knowns_all_prob, unknowns_all_prob),dim=0)
    all_predicted = torch.cat((knowns_all_predicted, unknowns_all_predicted),dim=0)

    OSE, knowns_accuracy, current_converage = eval.tensor_OSRC(all_gt, all_predicted, all_prob)
    ax_OSRC.plot(OSE, knowns_accuracy, color=colors[plot_count], label=args.label_names[plot_no])
    ax_coverage.plot(OSE, current_converage, color=colors[plot_count], label=args.label_names[plot_no])

    Precision, Recall, unique_scores = eval.calculate_binary_precision_recall(all_gt, all_predicted, all_prob)
    FScore = eval.F_score(Precision, Recall, ß=ß)
    ax_F1.plot(unique_scores, FScore, color=colors[plot_count], label=args.label_names[plot_no])
    ax_PR.plot(Recall, Precision, color=colors[plot_count], label=args.label_names[plot_no])
    plot_count+=1

    if args.heuristic_to_execute is not None:
        for heuristic_ in args.heuristic_to_execute:
            predicted, prob = heuristic.__dict__[heuristic_](all_prob, alpha=1)

            OSE, knowns_accuracy, current_converage = eval.tensor_OSRC(all_gt, predicted, prob)
            ax_OSRC.plot(OSE, knowns_accuracy, color=colors[plot_count], label=f"{args.label_names[plot_no]}_{heuristic_}")
            ax_coverage.plot(OSE, current_converage, color=colors[plot_count], label=f"{args.label_names[plot_no]}_{heuristic_}")

            Precision, Recall, unique_scores = eval.calculate_binary_precision_recall(all_gt, predicted, prob)
            FScore = eval.F_score(Precision, Recall, ß=ß)
            ax_F1.plot(unique_scores, FScore, color=colors[plot_count], label=f"{args.label_names[plot_no]}_{heuristic_}")
            ax_PR.plot(Recall, Precision, color=colors[plot_count], label=f"{args.label_names[plot_no]}_{heuristic_}")
            plot_count+=1

ax_OSRC.autoscale(enable=True, axis='x', tight=True)
ax_OSRC.set_ylim([0, 1])
ax_OSRC.set_xlim([0., 1.])
ax_OSRC.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
ax_OSRC.set_xlabel(f"False Positive Rate : Total Unknowns {unknowns_all_gt.shape[0]}", fontsize=18, labelpad=10)
figlegend.legend(ax_OSRC.get_legend_handles_labels()[0], ax_OSRC.get_legend_handles_labels()[1],loc='center', fontsize=18, frameon=False, ncol=2)
ax_OSRC.legend(loc='lower center', fontsize=6, frameon=False, bbox_to_anchor=(-0.1, 0.25), ncol=1)
fig_OSRC.savefig(args.output_dir / f"360_OSRC.pdf", bbox_inches="tight")
figlegend.savefig(args.output_dir / f"360_OSRC_legend.pdf", bbox_inches="tight")


ax_coverage.autoscale(enable=True, axis='x', tight=True)
ax_coverage.set_ylim([0, 1])
ax_coverage.set_xlim([0., 1.])
ax_coverage.set_ylabel('Coverage on knowns', fontsize=18, labelpad=10)
ax_coverage.set_xlabel(f"False Positive Rate : Total Unknowns {unknowns_all_gt.shape[0]}", fontsize=18, labelpad=10)
ax_coverage.legend(loc='lower center', fontsize=18, frameon=False, bbox_to_anchor=(-0.75, 0.25), ncol=1)
fig_coverage.savefig(args.output_dir / f"360_coverage.pdf", bbox_inches="tight")


ax_F1.set_ylim([0, 1])
ax_F1.set_xlim([0., 1.])
ax_F1.autoscale(enable=True, axis='x', tight=True)
ax_F1.set_ylabel(f'F-1 Score', fontsize=18, labelpad=10)
ax_F1.set_xlabel(f"Probability Scores", fontsize=18, labelpad=10)
ax_F1.legend(loc='lower center', fontsize=18, bbox_to_anchor=(-0.75, 0.25), frameon=False)
fig_F1.savefig(args.output_dir / f"360_F-1.pdf",
             bbox_inches="tight")


ax_PR.set_ylim([0, 1])
ax_PR.set_xlim([0., 1.])
ax_PR.autoscale(enable=True, axis='x', tight=True)
ax_PR.set_ylabel(f'Precision\nratio of detection that were knowns', fontsize=12, labelpad=10)
ax_PR.set_xlabel(f"Recall\nhow many knowns detected as knowns", fontsize=12, labelpad=10)
ax_PR.legend(loc='lower center', fontsize=18, bbox_to_anchor=(-0.75, 0.25), frameon=False)
fig_PR.savefig(args.output_dir / f"PR-curve.pdf",
             bbox_inches="tight")
