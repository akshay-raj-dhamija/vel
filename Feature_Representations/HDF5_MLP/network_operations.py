import torch
import torch.nn as nn
import numpy as np
import math
import copy
from vast import losses
import torch.utils.data as data_util
from torch.utils.tensorboard import SummaryWriter
from vast.tools import logger as vastlogger
torch.manual_seed(0)

logger = vastlogger.get_logger()


def get_loss_functions(args):
    approach = {"SoftMax": dict(first_loss_func=losses.nll_loss,
                                second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor([0.]).to(arg1.device)
                                ),
                "CenterLoss": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                                   second_loss_func=losses.tensor_center_loss(beta=0.1)
                                   ),
                "COOL": dict(first_loss_func=losses.entropic_openset_loss(num_of_classes=1000),
                             second_loss_func=losses.objecto_center_loss(
                                 args.Batch_Size,
                                 beta=0.1,
                                 classes=range(-1, 1000, 1),
                                 fc_layer_dimension=128,
                                 ring_size=args.Minimum_Knowns_Magnitude)
                             ),
                "BG": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                           second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor([0.]).to(arg1.device)
                           ),
                "entropic": dict(first_loss_func=losses.entropic_openset_loss(num_of_classes=1000),
                                 second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor([0.]).to(arg1.device)
                                 ),
                "objectosphere": dict(first_loss_func=losses.entropic_openset_loss(num_of_classes=1000),
                                      second_loss_func=losses.objectoSphere_loss(
                                          args.Batch_Size,
                                          knownsMinimumMag=args.Minimum_Knowns_Magnitude)
                                      )
                }
    return approach[args.approach].values()


class MLP(nn.Module):
    def __init__(self, input_feature_size=2048, num_classes=50, fc2_dim=64):
        super(MLP, self).__init__()
        self.fc2_dim = fc2_dim
        if self.fc2_dim is None:
            self.fc = nn.Linear(in_features=input_feature_size, out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=input_feature_size, out_features=fc2_dim, bias=True)
            self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=num_classes, bias=False)

    def forward(self, x):
        if self.fc2_dim is None:
            fc = self.fc(x)
            return fc, fc
        else:
            fc1 = self.fc1(x)
            fc2 = self.fc2(fc1)
            return fc2, fc1

def adjust_learning_rate(optimizer, epoch, total_epochs, original_lr):
    """Decay the learning rate based on schedule"""
    lr = original_lr
    for milestone in [90, 120, 150]:
    # for milestone in [90, 150]:
    # for milestone in [60, 80]:
        lr *= 0.1 if epoch >= milestone else 1.
    # lr = original_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class network():
    def __init__(self, num_classes, input_feature_size, output_dir = None, fc2_dim=64):
        self.net = MLP(num_classes=num_classes, input_feature_size=input_feature_size, fc2_dim=fc2_dim)
        self.net = self.net.cuda()
        logger.info(f"Model Architecture\n{self.net}")
        self.cls_names = []
        self.input_feature_size = input_feature_size
        self.output_dir = output_dir

    def prep_training_data(self, training_data, known_unknown_training_data):
        classes_in_consideration = sorted(list(training_data.keys()))
        training_tensor_x=[]
        training_tensor_label=[]
        training_tensor_y=[]
        for cls_no, cls in enumerate(sorted([*training_data])):
            training_tensor_x.append(training_data[cls])
            training_tensor_label.extend([cls]*training_data[cls].shape[0])
            training_tensor_y.extend([cls_no]*training_data[cls].shape[0])
        if known_unknown_training_data is not None:
            for cls in known_unknown_training_data:
                training_tensor_x.append(known_unknown_training_data[cls])
                training_tensor_label.extend([cls]*known_unknown_training_data[cls].shape[0])
                training_tensor_y.extend([-1]*known_unknown_training_data[cls].shape[0])
        training_tensor_x = torch.cat(training_tensor_x).type(torch.FloatTensor)
        training_tensor_label = np.array(training_tensor_label)
        training_tensor_y = torch.tensor(training_tensor_y).type(torch.LongTensor)
        # training_tensor_y=torch.zeros(training_tensor_label.shape[0]).type(torch.LongTensor)
        sample_weights = torch.ones(training_tensor_label.shape[0])
        sample_weights = sample_weights*1000.
        logger.debug(f"Training dataset size {list(training_tensor_x.shape)} "
                     f"labels size {list(training_tensor_y.shape)} "
                     f"sample weights size {list(sample_weights.shape)}")
        for cls_no,cls in enumerate(classes_in_consideration):
            sample_weights[training_tensor_label==cls]/=sample_weights[training_tensor_label==cls].shape[0]
        sample_weights[training_tensor_y == -1] /= sample_weights[training_tensor_y == -1].shape[0]
        training_tensor_x = training_tensor_x.cuda()
        training_tensor_y = training_tensor_y.cuda()
        sample_weights = sample_weights.cuda()
        second_sample_weights = torch.ones(training_tensor_x.shape[0]).cuda() * 1000
        second_sample_weights[training_tensor_y!=-1] = second_sample_weights[training_tensor_y!=-1]/\
                                                       training_tensor_y[training_tensor_y!=-1].shape[0]
        second_sample_weights[training_tensor_y==-1] = second_sample_weights[training_tensor_y==-1]/\
                                                       training_tensor_y[training_tensor_y==-1].shape[0]
        self.dataset = data_util.TensorDataset(training_tensor_x, training_tensor_y, sample_weights, second_sample_weights)
        self.cls_names = classes_in_consideration

    def training(self, training_data, known_unknown_training_data=None,
                 epochs=150, lr=0.01, batch_size=256, args=None):
        self.prep_training_data(training_data, known_unknown_training_data)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        loader = data_util.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        first_loss_func, second_loss_func = get_loss_functions(args)
        no_of_print_statements = min(10,epochs)
        printing_interval = epochs//no_of_print_statements
        summary_writer = None
        if self.output_dir is not None:
            summary_writer = SummaryWriter(f"{self.output_dir}/MLP_training_logs")
        best_accuracy = 0.
        best_loss = 1e+11
        best_model = None
        for epoch in range(epochs):
            loss_history=[]
            first_loss_history=[]
            second_loss_history=[]
            train_accuracy = torch.zeros(2, dtype=int)
            adjust_learning_rate(optimizer = optimizer, epoch = epoch, total_epochs = epochs, original_lr = lr)
            for x, y, first_sample_weight, second_sample_weight in loader:
                optimizer.zero_grad()
                output, features = self.net(x)
                first_loss = first_loss_func(output, y)
                second_loss = second_loss_func(features, y)
                if epoch<=50:
                    loss = (first_sample_weight * first_loss)
                else:
                    loss = (first_sample_weight * first_loss) + args.second_loss_weight*(second_sample_weight*second_loss)
                train_accuracy += losses.accuracy(output, y)
                loss_history.extend(loss.tolist())
                first_loss_history.extend(first_loss.tolist())
                second_loss_history.extend(second_loss.tolist())
                loss.mean().backward()
                optimizer.step()

            acc = float(train_accuracy[0]) / float(train_accuracy[1])
            to_print=f"Epoch {epoch:03d}/{epochs:03d} \t"\
                     f"train-loss: {np.mean(loss_history):1.5f}  \t"\
                     f"first-loss: {np.mean(first_loss_history):1.5f}  \t"\
                     f"second-loss: {np.mean(second_loss_history):1.5f}  \t"\
                     f"accuracy: {acc:9.5f}"
            if summary_writer is not None:
                summary_writer.add_scalar(f"{len(self.cls_names)}/loss",
                                          np.mean(loss_history), epoch)
                summary_writer.add_scalar(f"{len(self.cls_names)}/accuracy",
                                          float(train_accuracy[0])/float(train_accuracy[1]), epoch)
            if epoch%printing_interval==0:
                logger.info(to_print)
            else:
                logger.debug(to_print)
            if best_accuracy<=acc:
                best_model = copy.deepcopy(self.net.state_dict())

        # Replace network weights with best model weights
        self.net.load_state_dict(best_model)

    def inference(self, validation_data):
        results = {}
        for cls in validation_data:
            with torch.no_grad():
                logits, features = self.net(validation_data[cls].type(torch.FloatTensor).cuda())
                results[cls] = torch.nn.functional.softmax(logits, dim=1).cpu()
        logger.info(f"Inference done on {len(results)} classes")
        return results