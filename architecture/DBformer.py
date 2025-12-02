import sys
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
# from torchvision.models import resnet18, resnet50
from utils.utils import shuffle_batch, shuffle_instance
from architecture.transformer import Transformer, Transformer1, Transformer2, Transformer3, Transformer4, pos_enc_1d
import time
import timm
from architecture.abmilx import DAttentionX
from architecture.abmil import DAttention
# from django.conf import settings
from architecture.transmil import TransMIL

class CustomEncoder(nn.Module):
    def __init__(self, enc_type, pretrained):
        super(CustomEncoder, self).__init__()
        self.encoder = timm.create_model(enc_type, pretrained=pretrained, features_only=True)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.encoder(x)
        out = self.pool(features[-1]).squeeze(-1).squeeze(-1)
        return out

class DBformer(nn.Module):
    """
    Net that runs all the main components:
    patch encoder, dbps, patch aggregator and classification head
    """

    def get_conv_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):
        # Get architecture for patch encoder
        if enc_type == 'resnet18':
            res_net_fn = resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
            print("resnet50")
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None

        res_net = res_net_fn(weights=weights)
        # res_net = res_net_fn(pretrained=True)
        if n_chan_in == 1:
            # Standard resnet uses 3 input channels
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Compose patch encoder
        layer_ls = []
        layer_ls.extend([
            res_net.conv1,
            res_net.bn1,
            res_net.relu,
            res_net.maxpool,
            res_net.layer1,
            res_net.layer2
        ])

        if n_res_blocks == 4:
            layer_ls.extend([
                res_net.layer3,
                res_net.layer4
            ])
        layer_ls.append(res_net.avgpool)
        if (enc_type == 'resnet50'):
            layer_ls.append(self.get_projector2(2048, self.D))

        return nn.Sequential(*layer_ls)

    def get_large_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):

        encoder = CustomEncoder(enc_type, pretrained)

        encoder_output_dim = encoder.encoder.feature_info[-1]['num_chs'] 

        projector = self.get_projector(encoder_output_dim, self.D)

        large_model = nn.Sequential(
            encoder,
            projector
        )

        return large_model


    def get_projector2(self, n_chan_in, D):
        return nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            nn.LayerNorm(n_chan_in, eps=1e-05, elementwise_affine=False),
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )

    def get_projector1(self, n_chan_in, D):
        return nn.Sequential(
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )

    def get_projector(self, n_chan_in, D):
        return nn.Sequential(
            nn.LayerNorm(n_chan_in, eps=1e-05, elementwise_affine=False),
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )

    def get_output_layers(self, tasks):
        """
        Create an output layer for each task according to task definition
        """
        D = self.D
        n_class = self.n_class

        output_layers = nn.ModuleDict()
        for task in tasks.values():
            if task['act_fn'] == 'softmax':
                act_fn = nn.Softmax(dim=-1)
            elif task['act_fn'] == 'sigmoid':
                act_fn = nn.Sigmoid()

            layers = [
                nn.Linear(D, n_class),
                act_fn
            ]
            output_layers[task['name']] = nn.Sequential(*layers)

        return output_layers

    def __init__(self, device, conf):
        super().__init__()

        self.device = device
        self.n_class = conf.n_class
        self.M = conf.M
        self.test_M=conf.test_M
        self.I = conf.I
        self.D = conf.D
        self.S = conf.S
        self.use_pos = conf.use_pos
        self.tasks = conf.tasks
        self.shuffle = conf.shuffle
        self.sample = conf.sample_style
        self.shuffle_style = conf.shuffle_style
        self.is_image = conf.is_image
        self.lrs=conf.lrs
        self.single=conf.single

        if self.is_image:
            if("resnet" in conf.enc_type):
                self.encoder = self.get_conv_patch_enc(conf.enc_type, conf.pretrained,
                                                   conf.n_chan_in, conf.n_res_blocks)
            else:
                self.encoder = self.get_large_patch_enc(conf.enc_type, conf.pretrained,
                                                       conf.n_chan_in, conf.n_res_blocks)
        else:
            self.encoder = self.get_projector(conf.n_chan_in, self.D)

        # Define the multi-head cross-attention transformer
        if (conf.attention == "DA"):
            self.transf = Transformer4(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
                                   conf.D_inner, conf.attn_dropout, conf.dropout)
        if (conf.attention == "ABM"):
            self.transf = DAttentionX(
                input_dim=conf.D,
                n_classes=conf.n_class,
                dropout=conf.dropout,
                act="gelu",
                inner_dim=conf.D,
                n_heads=conf.H, 
                attn_plus=False,
                ffn=True,
                attn_dropout=conf.attn_dropout,
                proj_drop=conf.dropout
            )
        if (conf.attention == "trans"):
            self.transf = TransMIL(
                input_dim=conf.D,
                n_classes=conf.n_class,
                dropout=conf.dropout,
                act="gelu",
                inner_dim=conf.D,
                n_heads=conf.H,  
                pos = 'none',
            )
        if (conf.attention == "AB"):
            self.transf = DAttention(
                input_dim=conf.D,
                n_classes=conf.n_class,
                dropout=conf.dropout,
                act="gelu",
                inner_dim=conf.D,
            )
        if (conf.attention == "CA"):
            self.transf = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
                                  conf.D_inner, conf.attn_dropout, conf.dropout)
        if (conf.attention == "ABX"):
            self.transf = DAttentionX(
                input_dim=conf.D,
                n_classes=conf.n_class,
                dropout=conf.dropout,
                act="gelu",
                inner_dim=conf.D,
                n_heads=conf.H,
                attn_plus=True,
                ffn=True,
                attn_dropout=conf.attn_dropout,
                proj_drop=conf.dropout
            )

        # Optionally use standard 1d sinusoidal positional encoding
        if conf.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0).to(device)
        else:
            self.pos_enc = None

        # Define an output layer for each task
        self.output_layers = self.get_output_layers(conf.tasks)

    def do_shuffle(self, patches, pos_enc):
        """
        Shuffles patches and pos_enc so that patches that have an equivalent score
        are sampled uniformly
        """

        shuffle_style = self.shuffle_style
        if shuffle_style == 'batch':
            patches, shuffle_idx = shuffle_batch(patches)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_batch(pos_enc, shuffle_idx)
        elif shuffle_style == 'instance':
            patches, shuffle_idx = shuffle_instance(patches, 1)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_instance(pos_enc, 1, shuffle_idx)

        return patches, pos_enc

    def score_probability_select(self, emb, emb_pos, M, idx):
        """
        Scores embeddings and selects the top-M embeddings
        """
        B = emb.shape[0]
        N = emb.shape[1]
        D = emb.shape[2]

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb


        probabilities = self.transf.get_scores(emb_to_score).view(B,N)  # (B, N)
        top_idx = torch.multinomial(probabilities, M, replacement=False)  # (B, M)

        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        mem_idx = torch.gather(idx, 1, top_idx)

        selected_mask = torch.zeros((B,N), device=idx.device, dtype=torch.bool)
        selected_mask.scatter_(1, top_idx, True)
        sub_idx = idx[selected_mask.logical_not()].reshape(B, -1)
        sub_emb = torch.gather(emb, 1, sub_idx.unsqueeze(-1).expand(-1, -1, D))

        # print(sub_idx[0,0:5])
        # print(sub_idx[1, 0:5])
        # print(sub_idx[2, 0:5])
        # print()


        return mem_emb, mem_idx,sub_emb, attn

    def score_hard_select(self, emb, emb_pos, M, idx):
        """
        Scores embeddings and selects the top-M embeddings
        """
        B = emb.shape[0]
        N = emb.shape[1]
        D = emb.shape[2]


        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        attn = self.transf.get_scores(emb_to_score).view(B,N)  # (B, M+I)
        # print("维度 N 的所有数值:", attn[0, :].tolist())
        top_idx = torch.topk(attn, M, dim=-1)[1]  # (B, M)

        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        mem_idx = torch.gather(idx, 1, top_idx)

        selected_mask = torch.zeros((B, N), device=idx.device, dtype=torch.bool)
        selected_mask.scatter_(1, top_idx, True)
        sub_idx = idx[selected_mask.logical_not()].reshape(B, -1)
        sub_emb = torch.gather(emb, 1, sub_idx.unsqueeze(-1).expand(-1, -1, D))

        return mem_emb, mem_idx, sub_emb, attn

    def score_half_select(self, emb, emb_pos, M, idx):
        """
        Scores embeddings and selects the top-M embeddings
        """
        B = emb.shape[0]
        N = emb.shape[1]
        D = emb.shape[2]

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        attn = self.transf.get_scores(emb_to_score).view(B,N)  # (B, N)

        # attn = self.transf.get_scores(emb_to_score, return_attn=True)  # [B, H, N, N]
        # attn = attn.mean(dim=1).mean(dim=-1)  # [B, N]

        # print("维度 N 的所有数值:", attn[:, 1].tolist())
        top_idx = torch.topk(attn, M, dim=-1)[1]  # (B, M)

        probabilities1= torch.gather(attn, 1, top_idx) # (B, M)
        probabilities1 = torch.softmax(probabilities1, dim=-1) # (B, M)

        M1 = (M + 1) // 2
        mem_idx1 = torch.multinomial(probabilities1, M1, replacement=False)  # (B, M1)，相对idx
        mem_idx1 = torch.gather(top_idx, 1, mem_idx1) #(B, M1)，相对idx转绝对idx
        mem_emb1 = torch.gather(emb, 1, mem_idx1.unsqueeze(-1).expand(-1, -1, D))  # (B, M1, D)

        selected_mask = torch.zeros((B, N), device=idx.device, dtype=torch.bool)
        selected_mask.scatter_(1, mem_idx1, True)
        other_idx = idx[selected_mask.logical_not()].reshape(B, -1) # (B, N-M1)，绝对idx
        probabilities2 = torch.gather(attn, 1, other_idx)  # (B, N-M1)
        probabilities2 = torch.softmax(probabilities2, dim=-1)  # (B, N-M1)
        M2 = M - M1
        mem_idx2 = torch.multinomial(probabilities2, M2, replacement=False)  # (B, M2)，相对idx
        mem_idx2 = torch.gather(other_idx, 1, mem_idx2)  # (B, M2)，相对idx转绝对idx
        mem_emb2 = torch.gather(emb, 1, mem_idx2.unsqueeze(-1).expand(-1, -1, D))  # (B, M2, D)

        mem_emb = torch.cat([mem_emb1,mem_emb2],dim=1) # (B, M)
        mem_idx = torch.cat([mem_idx1, mem_idx2], dim=1) # (B, M)

        selected_mask = torch.zeros((B, N), device=idx.device, dtype=torch.bool)
        selected_mask.scatter_(1, mem_idx, True)
        sub_idx = idx[selected_mask.logical_not()].reshape(B, -1)  # (B, N-M)，绝对idx
        sub_emb = torch.gather(emb, 1, sub_idx.unsqueeze(-1).expand(-1, -1, D))




        return mem_emb, mem_idx, sub_emb, attn

    def get_preds(self, embeddings):
        preds = {}
        for task in self.tasks.values():
            t_name, t_id = task['name'], task['id']
            layer = self.output_layers[t_name]

            emb = embeddings[:, t_id]
            preds[t_name] = layer(emb)

        return preds

    @torch.no_grad()
    def no_gradient_encode(self,patches):
        if self.training:
            self.encoder.eval()
            self.transf.eval()
        
        emb=self.encoder(patches)
        
        if self.training:
            self.encoder.train()
            self.transf.train()
        
        return emb
    

    @torch.no_grad()
    def dbps(self, patches_input):
        """ Iterative Patch Selection """

        M = self.M
        I = self.I
        D = self.D
        S = self.S
        device = self.device
        shuffle = self.shuffle
        use_pos = self.use_pos
        pos_enc = self.pos_enc

        if use_pos:
            pos_enc = pos_enc.expand(B, -1, -1)
        if shuffle:
            patches_input, pos_enc = self.do_shuffle(patches_input, pos_enc)
        if (self.lrs):
            B, N, C, H, W = patches_input.shape
            patches = patches_input.view(B * N, C, H, W)
            patches = torch.nn.functional.interpolate(
                patches, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            patches = patches.view(B, N, C, patches.shape[-2], patches.shape[-1])
        else:
            patches = patches_input
        
        
        patch_shape = patches.shape
        patch_input_shape = patches_input.shape
        B, N = patch_shape[:2]

        # Shortcut: dbps not required when memory is larger than total number of patches
        if M >= N:
            # Batchify pos enc
            pos_enc = pos_enc.expand(B, -1, -1) if use_pos else None
            return patches.to(device), pos_enc

            # dbps runs in evaluation mode
        if self.training:
            self.encoder.eval()
            self.transf.eval()


        # Init memory buffer
        # Put patches onto GPU in case it is not there yet (lazy loading).
        # `to` will return self in case patches are located on GPU already (eager loading)
        init_patch = patches[:, :M].to(device)

        ## Embed
        mem_emb = self.encoder(init_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)

        # Init memory indixes in order to select patches at the end of dbps.
        idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = idx[:, :M]

        # Apply dbps for `n_iter` iterations
        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            # Get next patches
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)
            iter_patch = patches[:, start_idx:end_idx].to(device)
            iter_idx = idx[:, start_idx:end_idx]
            iter_emb = self.encoder(iter_patch.reshape(-1, *patch_shape[2:]))
            iter_emb = iter_emb.view(B, -1, D)
            mem_emb = torch.cat((mem_emb, iter_emb), dim=1)
            mem_idx = torch.cat((mem_idx, iter_idx), dim=1)
        if self.training:
            if(self.sample=="Half"):
                mem_emb, mem_idx,sub_emb, attn = self.score_half_select(mem_emb, None, M, mem_idx)
            elif (self.sample == "Hard"):
                mem_emb, mem_idx, sub_emb, attn = self.score_hard_select(mem_emb, None, M, mem_idx)
            elif (self.sample == "Prob"):
                mem_emb, mem_idx, sub_emb, attn = self.score_probability_select(mem_emb, None, M, mem_idx)
        else:
            mem_emb, mem_idx, sub_emb, attn = self.score_hard_select(mem_emb, None, M, mem_idx)
        # sub_emb = mem_emb[:, M:]
        # mem_emb, mem_idx = mem_emb[:, :M], mem_idx[:, :M]


        # Select patches
        # whole_emb, large_idx = second_emb, second_idx
        n_dim_expand = len(patch_shape) - 2

        mem_patch = torch.gather(patches_input, 1,
                                 mem_idx.view(B, -1, *(1,) * n_dim_expand).expand(-1, -1, *patch_input_shape[2:]).to(
                                     patches_input.device)
                                 ).to(device)

        if use_pos:
            mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            mem_pos = None

        # Set components back to training mode
        # Although components of `self` that are relevant for dbps have been set to eval mode,
        # self is still in training mode at training time, i.e., we can use it here.
        if self.training:
            self.encoder.train()
            self.transf.train()
        # Return selected patch and corresponding positional embeddings
        # print(mem_patch.shape)
        return mem_patch, mem_pos, sub_emb, mem_idx, mem_idx, attn

    def forward(self, mem_patch, mem_pos=None, whole_emb=None, mem_idx=None, large_idx=None,return_attention=False):
        """
        After M patches have been selected during dbps, encode and aggregate them.
        The aggregated embedding is input to a classification head.
        """
        patch_shape = mem_patch.shape
        B, M = patch_shape[:2]
        mem_emb = self.encoder(mem_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)
        D = mem_emb.shape[-1]
        if (self.single):
            input = mem_emb
        else:
            input = torch.cat([mem_emb, whole_emb], dim=1)
        if (return_attention == False):
            image_emb = self.transf(input).view(B, 1, D)
        else:
            image_emb, student_attention = self.transf(input, return_attention=return_attention)
            image_emb = image_emb.view(B, 1, D)
        preds = self.get_preds(image_emb)
        if(return_attention==False):
            return preds
        else:
            return preds, student_attention