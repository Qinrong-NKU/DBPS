#!/usr/bin/env python

import os
import yaml
from pprint import pprint
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.utils import Logger, Struct, save_checkpoint
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from data.traffic.traffic_dataset import TrafficSigns
from data.camelyon.camelyon_dataset1 import CamelyonFeatures1
from data.CQU_bpdd import MyDataset3
from architecture.ips_net import IPSNet
from training.iterative1 import train_one_epoch, evaluate
import random

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset = 'camelyon'# either one of {'mnist', 'camelyon', 'traffic','cqu_bpdd','fmow'}


# get config
with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print("Used config:"); pprint(c);
    conf = Struct(**c)
if conf.wandb:
    wandb.init(project=conf.project, config=conf,entity="xx",name=conf.title)

# fix the seed for reproducibility
torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
torch.cuda.manual_seed_all(conf.seed)

# define datasets and dataloaders
if dataset == 'mnist':
    train_data = MegapixelMNIST(conf, train=True)
    test_data = MegapixelMNIST(conf, train=False)
elif dataset == 'traffic':
    train_data = TrafficSigns(conf, train=True)
    test_data = TrafficSigns(conf, train=False)
elif dataset == 'camelyon':
    train_data = CamelyonFeatures1(conf, train=True)
    test_data = CamelyonFeatures1(conf, train=False)
elif dataset == 'cqu_bpdd':
    train_data = MyDataset3(conf, _type='train')
    #val_data = MyDataset(conf, _type='val')0.
    test_data = MyDataset3(conf, _type='test')
    #val_loader = DataLoader(val_data, batch_size=conf.B_seq, shuffle=False,
    #num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)
elif dataset == 'fmow':
    train_data = MyDataset3(conf, _type='train')
    test_data = MyDataset3(conf, _type='test')

train_loader = DataLoader(train_data, batch_size=conf.B_seq, shuffle=True
    ,num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)

test_loader = DataLoader(test_data, batch_size=conf.B_seq, shuffle=False
    ,num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)

# define network
net = IPSNet(device, conf).to(device)

loss_nll = nn.NLLLoss().cuda()
loss_bce = nn.BCELoss().cuda()

# define optimizer, lr not important at this point
optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=conf.wd)

criterions = {}
for task in conf.tasks.values():
    criterions[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce

log_writer_train = Logger(conf.tasks)
log_writer_test = Logger(conf.tasks)

best_acc, best_f1 = 0,0

for epoch in range(conf.n_epoch):
    train_one_epoch(net, criterions, train_loader, optimizer, device, epoch, log_writer_train, conf)

    log_writer_train.compute_metric()

    more_to_print = {'lr': optimizer.param_groups[0]['lr']}
    log_writer_train.print_stats(epoch, train=True, **more_to_print)

    #evaluate(net, criterions, val_loader, device, log_writer_test, conf)
    evaluate(net, criterions, test_loader, device, log_writer_test, conf)


    log_writer_test.compute_metric()
    log_writer_test.print_stats(epoch, train=False)

    acc = log_writer_test.metrics['metastases'][epoch]

    is_best = acc > best_acc and epoch > 0
    best_acc = max(acc, best_acc)
    if conf.wandb:
        wandb.log({
            'acc1': acc,
            'epoch': epoch,
        })

    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec1': best_acc,
            }, is_best,dir=conf.save_dir,title=conf.title)

# test
os.remove(conf.save_dir+conf.title+'_checkpoint.pth.tar')
best_model_dir = conf.save_dir+conf.title+'_model_best.pth.tar'
checkpoint = torch.load(best_model_dir, map_location = 'cpu')
net.load_state_dict(checkpoint['state_dict'])
print("<<<<<<<<<< Testing <<<<<<<<<<<<<")
evaluate(net, criterions, test_loader, device, log_writer_test, conf)
log_writer_test.compute_metric()
log_writer_test.print_stats(epoch, train=False)
if conf.wandb:
    wandb.log({
            'test_acc1': log_writer_test.metrics['metastases'][-1],
        })