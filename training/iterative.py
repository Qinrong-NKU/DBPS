import sys
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils.utils import adjust_learning_rate

class EntropyAlignmentLoss(nn.Module):
    def __init__(self, eps=1e-8):
        """
        Args:
            eps: Coefficient for numerical stability
            mode: Loss calculation mode (mse|huber|cosine)
        """
        super().__init__()
        self.eps = eps

    def _compute_entropy(self, x):
        """Calculate entropy of a normalized distribution"""
        log_x = torch.log(x + self.eps)
        entropy = -torch.sum(x * log_x, dim=-1)
        return entropy

    def forward(self, attn1, attn2, mode="none_mode"):
        """
        Args:
            attn1: [batch, seq_len] Normalized attention distribution 1
            attn2: [batch, seq_len] Normalized attention distribution 2
        Returns:
            loss: Scalar loss value
        """
        # Entropy calculation
        attn1 = attn1.squeeze(1)
        attn2 = attn2.squeeze(1)
        ent1 = self._compute_entropy(attn1)  # [B]
        ent2 = self._compute_entropy(attn2)  # [B]

        # Loss calculation
        if mode == 'mse':
            loss = F.mse_loss(ent1, ent2)
        elif mode == 'huber':
            loss = F.huber_loss(ent1, ent2)
        elif mode == 'cosine':
            loss = 1 - F.cosine_similarity(ent1.unsqueeze(0), ent2.unsqueeze(0))
        elif mode == 'mae':
            loss = F.l1_loss(ent1, ent2)
        elif mode == "max":
            loss = F.relu(ent1 - ent2).mean()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return loss


entropy = EntropyAlignmentLoss()


def init_batch(device, conf):
    """
    Initialize the memory buffer for the batch consisting of M patches
    """
    M1 = conf.test_M
    if conf.is_image:
        mem_patch = torch.zeros((conf.B, M1, conf.n_chan_in, *conf.patch_size)).to(device)
    else:
        mem_patch = torch.zeros((conf.B, M1, conf.n_chan_in)).to(device)

    if conf.use_pos:
        mem_pos_enc = torch.zeros((conf.B, M1, conf.D)).to(device)
    else:
        mem_pos_enc = None

    # Initialize labels for the batch (for multi-task scenarios like MNIST)
    labels = {}
    for task in conf.tasks.values():
        if task['metric'] == 'multilabel_accuracy':
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B,), dtype=torch.int64).to(device)

    return mem_patch, mem_pos_enc, labels


def fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
               mem_patch_iter, mem_pos_enc_iter, conf):
    """
    Fill the patch, position encoding and label buffers, and update helper variables
    """
    n_seq, len_seq = mem_patch_iter.shape[:2]
    mem_patch[n_prep:n_prep + n_seq, :len_seq] = mem_patch_iter
    if conf.use_pos:
        mem_pos_enc[n_prep:n_prep + n_seq, :len_seq] = mem_pos_enc_iter

    for task in conf.tasks.values():
        labels[task['name']][n_prep:n_prep + n_seq] = data[task['name']]

    n_prep += n_seq
    n_prep_batch += 1

    batch_data = (mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch)
    return batch_data


def shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf, whole_emb, mem_idx, large_idx):
    """
    Adjust batch by removing empty instances (may occur in the last batch of an epoch)
    """
    mem_patch = mem_patch[:n_prep]
    whole_emb = whole_emb[:n_prep]
    mem_idx = mem_idx[:n_prep]
    large_idx = large_idx[:n_prep]
    if conf.use_pos:
        mem_pos_enc = mem_pos_enc[:n_prep]

    for task in conf.tasks.values():
        labels[task['name']] = labels[task['name']][:n_prep]

    return mem_patch, mem_pos_enc, labels, whole_emb, mem_idx, large_idx


def compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf, whole_emb, mem_idx, large_idx):
    """
    Obtain model predictions, compute losses for each task, and collect logging statistics
    """
    # Get model predictions
    # time1 = time.perf_counter()
    preds = net(mem_patch, mem_pos_enc, whole_emb, mem_idx, large_idx)
    # time2 = time.perf_counter()
    # print('Model forward pass time: %s milliseconds' % ((time2 - time1) * 1000))

    # Compute loss for each task and sum them up
    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
    for task in conf.tasks.values():
        t_name, t_act = task['name'], task['act_fn']

        criterion = criterions[t_name]
        label = labels[t_name]
        pred = preds[t_name].squeeze(-1)
        
        if t_act == 'softmax':
            pred_loss = torch.log(pred + conf.eps)
            label_loss = label
        else:
            pred_loss = pred.view(-1)
            label_loss = label.view(-1).type(torch.float32)
        
        task_loss = criterion(pred_loss, label_loss)
        task_losses[t_name] = task_loss.item()
        task_preds[t_name] = pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += task_loss
    
    # Average loss across all tasks
    loss /= len(conf.tasks.values())

    return loss, [task_losses, task_preds, task_labels]


def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    """
    Train the given network for one full epoch using the specified loss functions (criterions)
    """
    conf.I = int(conf.I1 * (1 - conf.ratio))
    net.I = conf.I
    conf.N = int(conf.N * (1 - conf.ratio))
    net.N = conf.N

    # Set network to training mode
    net.train()

    # Initialize helper variables
    n_prep, n_prep_batch = 0, 0  # Number of prepared images/batches
    mem_pos_enc = None
    start_new_batch = True

    times = []  # Only used for tracking efficiency statistics
    # Iterate through the data loader
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch to GPU if eager execution is enabled (default), otherwise keep on CPU
        # Data is a dictionary with keys `input` (image patches) and `{task_name}` (labels for each task)
        image_patches = data['input'].to(device) if conf.eager else data['input']

        # If starting a new batch, initialize buffers to store batch data
        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
            # Start timing if efficiency tracking is enabled
            if conf.track_efficiency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

        # Apply dbps (Image Patch Selection) to input patches
        if conf.isParallel:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, _ = net.module.dbps(image_patches)
        else:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, _ = net.dbps(image_patches)

        # Fill batch buffers with output from dbps step
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        # Check if current batch is full or if it's the last batch of the epoch
        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        # Perform training step when batch is full or when processing the last batch
        if batch_full or is_last_batch:
            if not batch_full:
                # Trim empty entries from the last (incomplete) batch
                mem_patch, mem_pos_enc, labels, whole_emb, mem_idx, large_idx = shrink_batch(
                    mem_patch, mem_pos_enc, labels, n_prep, conf, whole_emb, mem_idx, large_idx
                )

            # Adjust learning rate
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it + 1)
            # Reset gradients
            optimizer.zero_grad()

            # Calculate loss
            loss, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf, whole_emb, mem_idx, large_idx)
            task_losses, task_preds, task_labels = task_info

            # Backpropagate gradients and update model parameters
            loss.backward()
            optimizer.step()

            # Log timing and memory usage if efficiency tracking is enabled
            if conf.track_efficiency:
                end_event.record()
                torch.cuda.synchronize()
                if epoch == conf.track_epoch and data_it > 0 and not is_last_batch:
                    times.append(start_event.elapsed_time(end_event))
                    print("Time elapsed: ", times[-1])

            # Update training log
            log_writer.update(task_losses, task_preds, task_labels)

            # Reset helper variables for next batch
            n_prep = 0
            start_new_batch = True

    # Print efficiency statistics if tracking is enabled
    if conf.track_efficiency:
        if epoch == conf.track_epoch:
            print("Average time per batch: ", np.mean(times))

            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            print(f"Peak memory usage: {peak_bytes_requirement / 1024 ** 3:.4f} GB")

            print("TORCH.CUDA.MEMORY_SUMMARY: ", torch.cuda.memory_summary())
            sys.exit()


# Disable gradient computation during evaluation
@torch.no_grad()
def evaluate(net, criterions, data_loader, device, log_writer, conf):
    conf.N = conf.N1
    net.N = conf.N
    conf.I = conf.I1
    net.I = conf.I

    # Set network to evaluation mode
    net.eval()

    # Remaining logic follows a similar structure to the training loop
    n_prep, n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True

    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        # Apply dbps to input patches
        if conf.isParallel:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, _ = net.module.dbps(image_patches)
        else:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, _ = net.dbps(image_patches)

        # Fill batch buffers
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        # Check batch status
        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or is_last_batch:
            if not batch_full:
                # Trim empty entries from incomplete batch
                mem_patch, mem_pos_enc, labels, whole_emb, mem_idx, large_idx = shrink_batch(
                    mem_patch, mem_pos_enc, labels, n_prep, conf, whole_emb, mem_idx, large_idx
                )

            # Compute loss (no backpropagation in evaluation)
            _, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf, whole_emb, mem_idx, large_idx)
            task_losses, task_preds, task_labels = task_info

            # Update evaluation log
            log_writer.update(task_losses, task_preds, task_labels)

            # Reset helper variables
            n_prep = 0
            start_new_batch = True