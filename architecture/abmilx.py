from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models._efficientnet_blocks import DepthwiseSeparableConv
from timm.layers import create_conv2d,get_norm_act_layer

from .vit_mil import SDPA,Mlp,sdpa
from .emb_position import SINCOS,get_alibi_slope,get_alibi_bias,VisionRotaryEmbedding,rotate_half
from .mlp import EmbeddingHead

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    for m in module.modules():
        if hasattr(m,'init_weights'):
            m.init_weights()

def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size

class SwiGLUAttn(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm='ln',
            norm_2=False,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm_type = norm
        #self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(hidden_features)
            if norm_2:
                self.norm2 = nn.BatchNorm1d(1)
            else:
                self.norm2 = nn.Identity()
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(hidden_features)
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        assert len(x.shape) == 4

        B, H, N, C = x.shape
        attn = x.reshape(B*H,N,C)

        attn_gate = self.fc1_g(attn)
        attn = self.fc1_x(attn)
        attn = self.act(attn_gate) * attn
        attn = self.drop1(attn)
        #attn = self.norm(attn)
        if self.norm_type == 'bn':
            attn = self.norm1(attn.transpose(-1,-2)).transpose(-1,-2)
        else:
            attn = self.norm1(attn)
        attn = self.fc2(attn)
        if self.norm_type == 'bn':
            attn = self.norm2(attn.transpose(-1,-2)).transpose(-1,-2)
        else:
            attn = self.norm2(attn)
        attn = self.drop2(attn)

        return attn.reshape(B, H, N, 1)  

class MLPAttn(nn.Module):
    def __init__(self,L=128,D=None,norm='ln',bias=False,norm_2=False,dp=0.,k=1,act='gelu',gated=False,dropout=True):
        super(MLPAttn, self).__init__()

        D = D or L
        
        self.gated = gated

        if self.gated:
            self.attention_a = [
            nn.Linear(L, D,bias=bias),
            ]
            if act == 'gelu': 
                self.attention_a += [nn.GELU()]
            elif act == 'relu':
                self.attention_a += [nn.ReLU()]
            elif act == 'tanh':
                self.attention_a += [nn.Tanh()]
            elif act == 'swish':
                self.attention_a += [nn.SiLU()]

            self.attention_b = [nn.Linear(L, D,bias=bias),
                                nn.Sigmoid()]

            if dropout:
                self.attention_a += [nn.Dropout(0.25)]
                self.attention_b += [nn.Dropout(0.25)]

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)

            self.attention_c = nn.Linear(D, k,bias=bias)
        else:
            self.fc1 = nn.Linear(L, D,bias=bias)

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'swish':
            self.act = nn.SiLU()

        self.norm_type = norm
        self.dp = nn.Dropout(dp) if dp else nn.Identity()
        if norm == 'bn':
            self.norm1 = nn.BatchNorm1d(D)
            if norm_2:
                self.norm2 = nn.BatchNorm1d(1)
            else:
                self.norm2 = nn.Identity()
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(D)
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        if not self.gated:
            self.fc2 = nn.Linear(D, k,bias=bias)
    
    def forward(self,x):
        #assert len(x.shape) == 4
        if len(x.shape) == 3:
            B, N, C = x.shape
            attn = x
            x_shape_3 = True
        else:
            B, H, N, C = x.shape
            attn = x.reshape(B*H,N,C)
            x_shape_3 = False

        if self.gated:
            attn = self.attention_a(attn)
            b = self.attention_b(attn)
            attn = attn.mul(b)
            attn = self.attention_c(attn)
        else:
            attn = self.fc1(attn)
            if self.norm_type == 'bn':
                attn = self.norm1(attn.transpose(-1,-2)).transpose(-1,-2)
            else:
                attn = self.norm1(attn)
            attn = self.act(attn)
            attn = self.dp(attn)

            attn = self.fc2(attn)
            if self.norm_type == 'bn':
                attn = self.norm2(attn.transpose(-1,-2)).transpose(-1,-2)
            else:
                attn = self.norm2(attn)
            # 这里和常规mlp不同，这里的维度是1了，不能dp
            #attn = self.dp(attn)

        if x_shape_3:
            attn = attn.transpose(-1,-2).unsqueeze(-1)
        else:
            attn = attn.reshape(B, H, N, 1)
        return attn        

class ConvAttn(nn.Module):
    def __init__(self,dim=128,k=7,norm='ln',bias=True,conv_norm_2=True,_type='dw'):
        super(ConvAttn, self).__init__()
        self._type = _type
        self.k = k
        self.norm_type = norm
        if _type == 'dw':
            groups = num_groups(1, dim)

            self.conv_dw = create_conv2d(
                dim, dim, k,
                stride=1,
                padding='same', groups=groups,bias=bias)

            if norm == 'ln':
                norm_layer = 'layernorm2d'
                assert not conv_norm_2
                norm_act_layer = get_norm_act_layer(norm_layer, nn.GELU)
                self.norm1 = norm_act_layer(dim, inplace=True)
            elif norm == 'bn':
                norm_layer = nn.BatchNorm2d
                norm_act_layer = get_norm_act_layer(norm_layer, nn.GELU)
                self.norm1 = norm_act_layer(dim, inplace=True)
            elif norm == 'none':
                assert not conv_norm_2
                self.norm1 = nn.GELU()
            else:
                raise NotImplementedError
            
            if conv_norm_2:
                self.norm2 = norm_act_layer(1, inplace=True, apply_act=False)
            else:
                self.norm2 = nn.Identity()

            self.conv_pw = create_conv2d(dim, 1, k, padding='same',bias=bias)
        
            #self.conv = DepthwiseSeparableConv(dim,1,k,1,pad_type='same',act_layer=nn.GELU,norm_layer=norm_layer)

        elif _type == 'naive':
            self.conv = create_conv2d(
                dim, 1, k,
                stride=1,
                padding='same', groups=1,bias=bias)
        elif _type == 'dw_1d':
            groups = num_groups(1, dim)

            self.conv_dw = nn.Conv1d(
                dim, dim, k,
                stride=1,
                padding='same', groups=groups,bias=bias)

            if norm == 'ln':
                assert not conv_norm_2
                self.norm1 = nn.LayerNorm(dim)
                self.act = nn.GELU()
            elif norm == 'bn':
                self.norm1 = nn.BatchNorm1d(dim)
                self.act = nn.GELU()
            elif norm == 'none':
                self.norm1 = nn.Identity()
                self.act = nn.Identity()
            else:
                raise NotImplementedError
            
            if conv_norm_2 and not norm == 'none':
                self.norm2 = nn.BatchNorm1d(1)
            else:
                self.norm2 = nn.Identity()

            self.conv_pw = nn.Conv1d(dim, 1, k, padding='same',bias=bias)
        else:
            raise NotImplementedError

    def forward_2d(self,x):

        _B, N, C = x.shape
         # padding
        _H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = _H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if _H < self.k:
            _H,W = self.k,self.k
            zero_pad = _H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((_B,zero_pad,C),device=x.device,dtype=x.dtype)],dim = 1)
            add_length += zero_pad
        
        #attn = self.conv(x_pad.transpose(1, 2).reshape(_B, C, _H, W)) # _B, 1, _H, W
        if self._type == 'dw':
            attn = self.conv_dw(x.transpose(1, 2).reshape(_B, C, _H, W))
            attn = self.norm1(attn)
            attn = self.conv_pw(attn)
            attn = self.norm2(attn)
        elif self._type == 'naive':
            attn = self.conv(x.transpose(1, 2).reshape(_B, C, _H, W))
        
        attn = attn.reshape(_B,_H * W)

        if add_length >0:
            attn = attn[:,:-add_length]

        return attn

    def forward(self,x):
        assert len(x.shape) == 4

        B, H, N, C = x.shape
        x_conv = x.reshape(B*H,N,C)

        if self._type == 'dw_1d':
            attn = self.conv_dw(x_conv.transpose(1, 2))
            if self.norm_type == 'ln':
                attn = self.norm1(attn.transpose(1, 2)).transpose(1, 2)
            else:
                attn = self.norm1(attn)
            attn = self.act(attn)
            attn = self.conv_pw(attn)
            attn = self.norm2(attn)
        else:
            attn = self.forward_2d(x_conv)

        attn = attn.reshape(B, H, N)

        return attn.unsqueeze(-1)

def check_tensor(tensor, tensor_name=""):
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaN values")
        raise ValueError
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains Inf values")
        raise ValueError
    if torch.isfinite(tensor).all():
        pass
        #print(f"{tensor_name} contains only finite values")

class AttnPlus(nn.Module):
    def __init__(self,dim=128,attn_dropout=0.,norm=True,embed=True,sdpa_type='torch',head=8,pos='',shortcut=True,v_embed=True,pad_v=False):
        super(AttnPlus, self).__init__()
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(attn_dropout)
        self.embed = embed
        self.sdpa_type = sdpa_type
        #self.sdpa = SDPA(sdpa_type=sdpa_type,head=head)
        self.shortcut = shortcut
        self.rope = None
        self.alibi = False
        self.pad_v = pad_v
        self.head = head
        self.dim = dim

        if embed:
            self.qk = nn.Linear(dim, dim * 2, bias = False)
            self.v = nn.Linear(1, 1, bias = False) if v_embed else nn.Identity()
        if norm:
            self.norm_x = nn.LayerNorm(dim)
        else:
            self.norm_x = nn.Identity()

        if 'alibi' in pos:
            self.alibi = True
            if 'learn' in pos:
                self.m = nn.Parameter(torch.ones(head))
            else:
                self.register_buffer("m", get_alibi_slope(head))
        elif 'rope' in pos:
            self.rope = VisionRotaryEmbedding(dim,online_rope=True,decouple=True,custom_freqs='ntk-aware')

    def forward(self,x,A,pos=None):
        if len(x.shape) == 3:
            B, N, _ = x.shape
            qk = self.qk(self.norm_x(x)).reshape(B, N, 2 ,self.head,self.dim // self.head).permute(2, 0, 3, 1, 4)
            q,k = qk.unbind(0)
        # B H N D
        else:
            B, H, N, D = x.shape
            qk = self.qk(self.norm_x(x)).reshape(B, self.head, N ,2 ,self.dim)
            q,k = qk.unbind(-2)
        #if self.embed:
        
        v = self.v(A)
        #else:
            #raise NotImplementedError
            #q,k = self.norm_x(x),self.norm_x(x)
            #v = A
        
        if self.rope is not None:

            freqs_cos, freqs_sin = self.rope.online_get_2d_rope_from_grid(pos[:,1:].transpose(-1,-2), pos[:,0][..., [1, 0]].unsqueeze(0))
            try:
                check_tensor(freqs_cos,'freqs_cos')
            except:
                # with open('../tmp/log', 'a') as f:
                #     # 使用print函数将张量值输出到文件
                #     [print(_pos,file=f) for _pos in pos]
                # print(pos)
                print(pos[:,1:].transpose(-1,-2))
                print(freqs_cos.shape)
                print(freqs_sin.shape)
                assert 1 == 2
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            freqs_cos,freqs_sin = freqs_cos.type(q.dtype),freqs_sin.type(q.dtype)
            q = q * freqs_cos + rotate_half(q) * freqs_sin
            k = k * freqs_cos + rotate_half(k) * freqs_sin

        # torch math做了pad之后，backward好像会变. torch 2.5.1
        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            # 维度补齐，方便使用flash_attn和memo_effi
            v = F.pad(v, (0, q.shape[-1] - v.size(-1)))
        A_plus = sdpa(
            q,k,v,
            attn_drop=self.attn_drop,
            scale=self.scale,
            training=self.training,
            sdpa_type=self.sdpa_type,
            attn_mask=get_alibi_bias(self.m,pos,device=q.device) if self.alibi else None
        ).transpose(1,2)

        if self.sdpa_type not in ('torch_math','torch') or self.pad_v:
            A_plus = A_plus[:,:,:,0].unsqueeze(-1)
        
        if self.shortcut:
            A = A + A_plus
        
        return A
        
class DAttentionX(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act,mil_norm=None,mil_bias=True,mil_cls_bias=True,inner_dim=512,embed_feat=True,feat_embed_type='norm',embed_feat_mlp_ratio=4,n_heads=6,proj_drop=0.,D=None,attn_type='mlp',conv_k=7,attn_norm='ln',attn_bias=True,attn_norm_2=True,conv_type='dw',attn_plus=False,ffn=False,attn_dropout=0.,ffn_bias=False,ffn_dp=0.,ffn_ratio=4.,sdpa_type='torch',pos=None,embed_norm_pos=0,no_attn_plus_test=False,attn_plus_sc=True,attn_plus_v_embed=True,pad_v=False,attn_vembed=False,attn_plus_embed_new=False,attn_act='gelu',**kwargs):
        super(DAttentionX, self).__init__()
        self.head_dim = inner_dim // n_heads
        self.L = inner_dim if attn_plus_embed_new else self.head_dim
        if D is None:
            self.D = self.L
        else:
            if D > 10.:
                self.D = int(D) if attn_plus_embed_new else int(D) // n_heads  
            else:
                self.D = int(self.L // D)
        #self.D = self.L if D is None else int(self.L // D) 

        self.K = 1
        self.feature = []
        self.mil_norm = mil_norm
        self.n_heads = n_heads
        self.attn_plus = attn_plus
        self.no_attn_plus_test = no_attn_plus_test
        self.pos = pos
        self.embed_norm_pos = embed_norm_pos
        self.attn_plus_embed_new = attn_plus_embed_new

        if mil_bias:
            mil_cls_bias = True

        assert pos in ('sincos','none','alibi','alibi_learn','rope',None)

        if pos == 'sincos':
            self.pos_embed = SINCOS()
        else:
            self.pos_embed = nn.Identity()

        if attn_plus:
            if pos is None:
                pos = ''
            self.attn_plus_fn = AttnPlus(inner_dim if attn_plus_embed_new else self.head_dim,sdpa_type=sdpa_type,head=n_heads,pos=pos,shortcut=attn_plus_sc,v_embed=attn_plus_v_embed,pad_v=pad_v)

        if ffn:
            self.norm_ffn = nn.LayerNorm(inner_dim) if not mil_norm else nn.Identity()
            self.mlp = Mlp(
            in_features=inner_dim,
            hidden_features=int(inner_dim * ffn_ratio),
            act_layer=nn.GELU,
            bias=ffn_bias,
            drop=ffn_dp,
            )
        self.ffn = ffn

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
            self.norm1 = nn.BatchNorm1d(self.L*self.K)
        elif mil_norm == 'ln':
            if embed_norm_pos == 0:
                self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
                self.norm1 = nn.LayerNorm(inner_dim,bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim,bias=mil_bias)
                self.norm1 = nn.LayerNorm(inner_dim,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()
        
        if embed_feat:
            if feat_embed_type == 'norm':
                self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                #self.feature += [nn.GELU()]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
            elif feat_embed_type == 'act_first':
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
            elif feat_embed_type == 'mlp':
                self.feature += [nn.Linear(input_dim, input_dim // embed_feat_mlp_ratio,bias=mil_bias)]
                if act.lower() == 'gelu':
                    self.feature += [nn.GELU()]
                elif act.lower() == 'relu':
                    self.feature += [nn.ReLU()]
                if dropout:
                    self.feature += [nn.Dropout(dropout)]
                self.feature += [nn.Linear(input_dim // embed_feat_mlp_ratio, inner_dim,bias=mil_bias)]
        else:
            if dropout:
                self.feature += [nn.Dropout(dropout)]
        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        if attn_type == 'mlp':
            self.attention = MLPAttn(self.L,self.D,norm=attn_norm,bias=attn_bias,norm_2=attn_norm_2,dp=attn_dropout,
            k=n_heads if attn_plus_embed_new else 1,
            act=attn_act)
        elif attn_type == 'mlp_gated':
            self.attention = MLPAttn(self.L,self.D,norm=attn_norm,bias=attn_bias,norm_2=attn_norm_2,dp=attn_dropout,
            k=n_heads if attn_plus_embed_new else 1,
            act=attn_act,gated=True)
        elif attn_type == 'swiglu':
            raise NotImplementedError
            self.attention = SwiGLUAttn(self.L,self.D,norm=attn_norm,norm_2=attn_norm_2,bias=attn_bias,drop=attn_dropout)
        elif attn_type == 'conv':
            raise NotImplementedError
            self.attention = ConvAttn(self.L,k=conv_k,norm=attn_norm,bias=attn_bias,conv_norm_2=attn_norm_2,_type=conv_type)

        if attn_vembed:
            self.attn_vembed = nn.Linear(inner_dim, inner_dim,bias=mil_bias) if attn_plus_embed_new else nn.Linear(self.head_dim, self.head_dim,bias=mil_bias)
        else:
            self.attn_vembed = nn.Identity()

        self.proj = nn.Linear(inner_dim, inner_dim,bias=mil_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(initialize_weights)
        
    def get_scores(self, x, return_attn=False,no_norm=False,return_act=False,pos=None,return_img_feat=False):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        if pos is not None:
            if len(pos.shape) == 2:
                pos = pos.unsqueeze(0)

        B, N, _ = x.shape

        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)

        if self.pos == 'sincos':
            x = self.pos_embed(x,pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        _,_,C = x.shape
        act = x.clone()

        if self.attn_plus_embed_new:
            A = self.attention(x)   # B N D
            x = self.attn_vembed(x)
        else:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D
            A = self.attention(x)   # B H N K
            x = self.attn_vembed(x)

        # ~~TODO：flash-attn 2.7.0不支持，q,k 和v的dim不同，但后续可能支持，应该持续关注，然后更新~~
        # 用pad的方式，解决dim不同的问题
        if self.attn_plus:
            if self.no_attn_plus_test and not self.training:
                pass
            else:
                A = self.attn_plus_fn(x,A,pos=pos)
        A = torch.transpose(A, -1, -2)  # B H K N
        A = F.softmax(A, dim=-1)  # softmax over N

        return A.mean(dim=1)
    
    def forward(self, x, return_attn=False,no_norm=False,return_act=False,pos=None,return_img_feat=False,return_attention=False):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        if pos is not None:
            if len(pos.shape) == 2:
                pos = pos.unsqueeze(0)

        B, N, _ = x.shape

        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)

        if self.pos == 'sincos':
            x = self.pos_embed(x,pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        _,_,C = x.shape
        act = x.clone()

        if self.attn_plus_embed_new:
            A = self.attention(x)   # B N D
            x = self.attn_vembed(x)
        else:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D
            A = self.attention(x)   # B H N K
            x = self.attn_vembed(x)

        # ~~TODO：flash-attn 2.7.0不支持，q,k 和v的dim不同，但后续可能支持，应该持续关注，然后更新~~
        # 用pad的方式，解决dim不同的问题
        if self.attn_plus:
            if self.no_attn_plus_test and not self.training:
                pass
            else:
                A = self.attn_plus_fn(x,A,pos=pos)
        A = torch.transpose(A, -1, -2)  # B H K N
        A = F.softmax(A, dim=-1)  # softmax over N

        if self.attn_plus_embed_new:
            x = x.reshape(B,N,self.n_heads,self.head_dim).transpose(1,2) # B H N D

        x = torch.einsum('b h k n, b h n d -> b h k d', A,x).squeeze(1) # B H D

        x = x.reshape(B, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.ffn:
            x = x + self.mlp(self.norm_ffn(x))

        x = self.norm1(x)
        if(return_attention==False):
            return x
        else:
            return x,A.mean(dim=1)
        # _logits = self.classifier(x)
        # 
        # if return_img_feat:
        #     _logits = [_logits,x]
        # 
        # if return_attn:
        #     output = []
        #     output.append(_logits)
        #     output.append(A.squeeze(-2))
        #     if return_act:
        #         output.append(act.squeeze(1))
        #     return output
        # else:   
        #     return _logits



