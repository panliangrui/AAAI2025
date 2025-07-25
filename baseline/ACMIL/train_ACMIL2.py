
# !/usr/bin/env python
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import ACMIL_GA
from architecture.transformer import ACMIL_MHA
import torch.nn.functional as F

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate,write_dict_to_csv
from timm.utils import accuracy
import torchmetrics
from datasets.base import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--n_token", type=int, default=5, help="number of attention branches in (MBA)."
    )
    parser.add_argument(
        "--n_masked_patch", type=int, default=10, help="top-K instances are be randomly masked in STKIM."
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0.6, help="maksing ratio in the STKIM"
    )
    parser.add_argument("--arch", type=str, default='ga', choices=['ga', 'mha'], help="choice of architecture type")
    parser.add_argument('--pretrain', default='medical_ssl',
                        choices=['natural_supervsied', 'medical_ssl', 'path-clip-L-336'],
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument("--lr", type=float, default=0.0001, help="maksing ratio in the STKIM")
    parser.add_argument( "--wd", type=float, default=0.00001, )
    parser.add_argument( "--warmup_epoch", type=float, default=0,)
    parser.add_argument( "--train_epoch", type=float, default=50, )
    parser.add_argument( "--B", type=int, default=1,)
    parser.add_argument("--D_feat",default=1536,type=int)
    parser.add_argument("--D_inner",default=512,type=int)
    parser.add_argument("--min_lr", type=int,default=0)
    parser.add_argument("--n_worker",default=0,type=int)
    parser.add_argument("--pin_memory",type=bool,default=False)



    args = parser.parse_args()
    return args
def test(dataname="CPTAC",seed=0):
    conf = get_arguments()
    conf.dataname = dataname
    conf.seed=seed

    set_data(conf)
    # Prepare dataset
    set_seed(conf.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    if conf.arch == 'ga':
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
    else:
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
    print("load model",conf.ckp)
    checkpoint=torch.load(conf.ckp)
    model.load_state_dict(checkpoint["model"])
    model.to(device)


    criterion = nn.CrossEntropyLoss()

    test_auc, test_acc, test_f1, test_loss,test_recall,test_pre = evaluate(model, criterion, test_loader, device, conf, 'Test')
    write_dict_to_csv(conf.csv_path,{"acc":test_acc,"auc":test_auc,"pre":test_pre,"recall":test_recall,"f1":test_f1},"w" if seed==0 else "a")

def main(dataname="CPTAC",seed=0):
    # Load config file
    conf = get_arguments()
    conf.dataname=dataname
    conf.seed=seed
    set_data(conf)

    # Prepare dataset
    set_seed(conf.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    if conf.arch == 'ga':
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
    else:
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch(model, criterion, train_loader, optimizer0, device, epoch, conf)


        val_auc, val_acc, val_f1, val_loss,val_recall,val_pre = evaluate(model, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss,test_recall,test_pre = evaluate(model, criterion, test_loader, device, conf, 'Test')


        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,is_best=True,seed=seed)
        print('\n')


    save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,is_last=True,seed=seed)
    print("Results on best epoch:")
    print(best_state)

def train_one_epoch(model, criterion, data_loader, optimizer0, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100


    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)

        # Compute loss
        sub_preds, slide_preds, attn = model(image_patches)
        if conf.n_token > 1:
            loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
        else:
            loss0 = torch.tensor(0.)

        loss1 = criterion(slide_preds, labels)


        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)
        # if conf.arch == 'mha':
        #     for i in range(8):
        #         for j in range(i + 1, 8):
        #             diff_loss += torch.cosine_similarity(attn[i], attn[j], dim=-1).mean() / (
        #                     8 * (8 - 1) / 2)

        for i in range(conf.n_token):
            for j in range(i + 1, conf.n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            conf.n_token * (conf.n_token - 1) / 2)
        
        loss = diff_loss + loss0 + loss1

        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()
        optimizer0.step()


        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(sub_loss=loss0.item())
        metric_logger.update(diff_loss=diff_loss.item())
        metric_logger.update(slide_loss=loss1.item())





# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):

    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)


        sub_preds, slide_preds, attn = net(image_patches)
        div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
        loss = criterion(slide_preds, labels)
        pred = torch.softmax(slide_preds, dim=-1)


        acc1 = accuracy(pred, labels, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.update(div_loss=div_loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred.append(pred)
        y_true.append(labels)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, average = 'macro',task=conf.task).to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()

    F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, average = 'macro',task=conf.task).to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()

    acc_metric=torchmetrics.Accuracy(num_classes=conf.n_class,average="macro",task=conf.task).to(device)
    acc_metric(y_pred, y_true)
    acc=acc_metric.compute().item()

    recall_metric=torchmetrics.Recall(num_classes=conf.n_class,average="macro",task=conf.task).to(device)
    recall_metric(y_pred, y_true)
    recall=recall_metric.compute().item()

    pre_metric=torchmetrics.Precision(num_classes=conf.n_class,average="macro",task=conf.task).to(device)

    pre_metric(y_pred, y_true)
    pre=pre_metric.compute().item()

    print('* Acc@1 {top1:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=acc, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, acc, f1_score, metric_logger.loss.global_avg,recall,pre



if __name__ == '__main__':

    for i in range(5):
        # main(dataname="tcga", seed=i)
        # test(dataname="tcga", seed=i)
        # main(dataname="cptac",seed=i)
        # test(dataname="cptac",seed=i)
        main(dataname="xiangya3",seed=i)
        test(dataname="xiangya3", seed=i)

