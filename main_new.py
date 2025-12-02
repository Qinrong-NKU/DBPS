#!/usr/bin/env python
import os
import yaml
import time
import random
import argparse  
from pprint import pprint
import numpy as np
import torch
import wandb
import swanlab
from torch import nn
from torch.utils.data import DataLoader

# Custom module imports
from utils.utils import Logger, Struct, save_checkpoint
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from data.traffic.traffic_dataset import TrafficSigns
# from data.camelyon.camelyon_dataset import CamelyonFeatures
from data.downstream_data import MyDataset
from architecture.DBformer import DBformer
from training.iterative import train_one_epoch, evaluate
from training.vis_iter_test import evaluate_attn_mask


def main():

    parser = argparse.ArgumentParser(description='the main')
    parser.add_argument('--dataset', type=str, required=False, 
                    default='cqu_bpdd')
    parser.add_argument('--config', type=str, required=False, 
                    default='new_config/cqu_bpdd_db+.yml', 
                    help='The path of the configuration file, for example:new_config/cqu_bpdd_db+.yml')
    args = parser.parse_args()

    # Device Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")


    # Configuration Loading 
    config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file does not exist: {config_path}")
    
    # Load and print configuration
    with open(config_path, "r") as ymlfile:
        config_dict = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print("Used config:")
        pprint(config_dict)
        conf = Struct(**config_dict)  # Convert to structured object for easy attribute access


    #  Logging Tools Initialization 
    if conf.wandb:
        wandb.init(
            project=conf.project,
            config=conf,
            entity="2603292100-leo",
            name=conf.title
        )
    
    if conf.swanlab:
        swanlab.init(
            workspace="329714",
            project=conf.project,
            config=config_path,
            experiment_name=conf.title
        )


    #  Random Seed Setup (for reproducibility) 
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True


    #  Dataset and DataLoader Setup 
    # Dataset mapping table (simplifies conditional checks)
    dataset_mapping = {
        'mnist': MegapixelMNIST,
        'traffic': TrafficSigns,
        # 'camelyon': CamelyonFeatures,
        'cqu_bpdd': MyDataset,
        'fmow': MyDataset,
        'eyes': MyDataset,
        'mame': MyDataset,
        'xbd': MyDataset
    }

    dataset = args.dataset

    # Initialize datasets
    DatasetClass = dataset_mapping[dataset]
    if dataset in ['mnist', 'traffic']:  # Datasets with special initialization
        train_data = DatasetClass(conf, train=True)
        test_data = DatasetClass(conf, train=False)
    else:  # General initialization (MyDataset)
        train_data = DatasetClass(conf, _type='train')
        test_data = DatasetClass(conf, _type='test')

    # Initialize data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=conf.B_seq,
        shuffle=True,
        num_workers=conf.n_worker,
        pin_memory=conf.pin_memory,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=conf.B_seq,
        shuffle=False,
        num_workers=conf.n_worker,
        pin_memory=conf.pin_memory,
        persistent_workers=True
    )


    #  Model and Training Components 
    # Initialize model
    net = DBformer(device, conf).to(device)
    if num_gpus > 1 and conf.isParallel:
        net = nn.DataParallel(net)  # Enable multi-GPU parallelism

    # Loss functions (selected based on task type)
    loss_nll = nn.NLLLoss().to(device)
    loss_bce = nn.BCELoss().to(device)
    criterions = {
        task['name']: loss_nll if task['act_fn'] == 'softmax' else loss_bce
        for task in conf.tasks.values()
    }

    # Optimizer (initial learning rate set to 0, may be adjusted dynamically)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=0,
        weight_decay=conf.wd
    )

    # Log recorders
    log_writer_train = Logger(conf.tasks)
    log_writer_test = Logger(conf.tasks)


    #  Training Process 
    best_acc, best_f1 = 0, 0
    if not conf.isOnlyTest:
        for epoch in range(conf.n_epoch):
            # Train for one epoch
            start_time = time.time()
            train_one_epoch(
                net, criterions, train_loader, 
                optimizer, device, epoch, 
                log_writer_train, conf
            )
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds.")

            # Calculate and print training logs
            log_writer_train.compute_metric()
            more_to_print = {'lr': optimizer.param_groups[0]['lr']}
            log_writer_train.print_stats(epoch, train=True,** more_to_print)

            # Evaluate on test set
            evaluate(net, criterions, test_loader, device, log_writer_test, conf)
            log_writer_test.compute_metric()
            log_writer_test.print_stats(epoch, train=False)

            # Record best metrics
            current_acc = log_writer_test.metrics['sign'][epoch]
            current_f1 = log_writer_test.f1
            is_best = current_acc > best_acc and epoch > 0
            best_acc = max(current_acc, best_acc)
            best_f1 = max(current_f1, best_f1)
            print(f'Current best accuracy: {best_acc}')

            # Log to monitoring tools
            if conf.wandb:
                wandb.log({'acc1': current_acc, 'f1': current_f1, 'epoch': epoch})
            if conf.swanlab:
                swanlab.log({
                    'acc1': current_acc, 
                    'f1': current_f1, 
                    'epoch': epoch,
                    'best_acc': best_acc
                })

            # Save model checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec1': best_acc,
                'best_f1': best_f1,
            }, is_best, dir=conf.save_dir, title=conf.title)

        # Clean up temporary checkpoint after training
        os.remove(os.path.join(conf.save_dir, f"{conf.title}_checkpoint.pth.tar"))
        best_model_path = os.path.join(conf.save_dir, f"{conf.title}_model_best.pth.tar")
    else:
        # In test-only mode: directly load specified model
        best_model_path = conf.test_dir


    #  Final Test Evaluation 
    print("<<<<<<<<<< Starting Final Testing <<<<<<<<<<<<<")
    # Load best model
    checkpoint = torch.load(
        best_model_path,
        map_location='cpu',
        weights_only=False
    )
    net.load_state_dict(checkpoint['state_dict'])

    # Evaluate (with attention visualization or normal evaluation)
    if conf.eval_with_attn:
        evaluate_attn_mask(net, criterions, test_loader, device, log_writer_test, conf)
    else:
        evaluate(net, criterions, test_loader, device, log_writer_test, conf)

    # Calculate and print test metrics
    log_writer_test.compute_metric()
    if not conf.isOnlyTest:
        log_writer_test.print_stats(epoch, train=False)

    # Record test results to logging tools
    test_acc = log_writer_test.metrics['sign'][-1]
    test_f1 = log_writer_test.f1
    print(f"Test accuracy: {test_acc}")
    print(f"Experiment title: {conf.title}")

    if conf.wandb:
        wandb.log({'test_acc1': test_acc, 'test_f1': test_f1})
    if conf.swanlab:
        swanlab.log({'test_acc1': test_acc, 'test_f1': test_f1})


if __name__ == "__main__":
    main()
