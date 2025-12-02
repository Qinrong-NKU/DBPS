import sys
import numpy as np
import torch
from PIL import Image
import matplotlib

def feature_imshow(inp, title=None):
    """Convert tensor to displayable image format"""
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

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

    # Initialize labels for the batch (for multi-task scenarios in MNIST)
    labels = {}
    for task in conf.tasks.values():
        if task['metric'] == 'multilabel_accuracy':
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B,), dtype=torch.int64).to(device)

    return mem_patch, mem_pos_enc, labels


@torch.no_grad()
def evaluate_attn_mask(net, criterions, data_loader, device, log_writer, conf):
    conf.N = conf.N1
    net.N = conf.N
    conf.I = conf.I1
    net.I = conf.I
    net.eval()
    start_new_batch = True
    original_number = 0

    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']
        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        

        if conf.isParallel:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, attn = net.module.dbps(image_patches)
        else:
            mem_patch_iter, mem_pos_enc_iter, whole_emb, mem_idx, large_idx, attn = net.dbps(image_patches)

        # Configuration for image grid reconstruction
        width_number = int(conf.width/conf.patch_size[1])
        height_number = int(conf.height/conf.patch_size[0])  
        total_iterations = width_number * height_number
        mask_list = []

        
        original_image = image_patches[0]
        
        min_value = torch.min(attn[0])
        max_value = torch.max(attn[0])
        normalized_attn = (attn[0] - min_value) / (max_value - min_value)

        # Generate attention-masked patches
        for num1 in range(total_iterations):
            attn_value = float(normalized_attn[num1])
            alpha = 0.5  
            

            
            patch = feature_imshow(original_image[num1].view(3, 200, 200))
            patch_with_mask = patch.copy()

            # Create red-to-blue heatmap for attention visualization
            cmap = matplotlib.cm.get_cmap('coolwarm')  
            heatmap = cmap(attn_value) 
            heatmap = heatmap[:3]  

            # Overlay heatmap on the original patch with transparency
            for c in range(3):  
                patch_with_mask[:, :, c] = patch[:, :, c] * (1 - alpha) + heatmap[c] * alpha

            mask_list.append(patch_with_mask)

        # Reconstruct full image from masked patches
        reconstructed_rows = []
        for number2 in range(height_number):
            # Assemble one row of patches
            row_patches = []
            for number3 in range(width_number):
                row_patches.append(mask_list[number2 * width_number + number3])
            row_combined = np.concatenate(row_patches, axis=1)  
            reconstructed_rows.append(row_combined)
        
        # Combine all rows to form the full masked image
        masked_image = np.concatenate(reconstructed_rows, axis=0)  
        # Convert to PIL Image (scale from [0,1] to [0,255])
        masked_image_pil = Image.fromarray(np.uint8(masked_image * 255)).convert('RGB')

        # Save the masked image
        import os
        save_dir = conf.eval_vis_path
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"masked_image_{original_number}.jpg")
        masked_image_pil.save(save_path)

        original_number += 1  
    
    sys.exit()  