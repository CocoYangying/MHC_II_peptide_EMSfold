#!/usr/bin/env python
"""
cv_fullmat_transformer.py
=========================
5-fold cross-validation training of the full-matrix Transformer model on MHC-peptide
ESMFold structural features.
Author: Yang Ying YYANG047@e.ntu.edu.sg
"""
import argparse, logging, time, math, os, glob
import pandas as pd, numpy as np, h5py, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, r2_score,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from scipy.stats import pearsonr
import csv
import os
from datetime import datetime
from datetime import timedelta 
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import functools
from tqdm import tqdm
import threading

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp import MixedPrecision
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_SOCKET_TIMEOUT'] = '1800'

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logging(save_dir, fold, rank):
    log_dir = os.path.join(save_dir, f"fold{fold}_logs")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_fold{fold}_{timestamp}.log")
    
    logger = logging.getLogger(f"fold{fold}_rank{rank}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler (only for rank 0)
    if rank == 0:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler (all ranks)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(f"Rank {rank} | %(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

import h5py, torch
from torch.utils.data import Dataset
class StableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.05, 0.05)
        nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected in StableLinear input")
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        
        x_clamped = torch.clamp(x, -5.0, 5.0)
        
        weight_clamped = torch.clamp(self.weight, -0.5, 0.5)
        bias_clamped = torch.clamp(torch.nan_to_num(self.bias, nan=0.0, posinf=10.0, neginf=-10.0), -1.0, 1.0)
        
        output = F.linear(x_clamped, weight_clamped, bias_clamped)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf detected in StableLinear output")
            output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return output


class StableLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        var = var + self.eps
        x_normalized = (x - mean) / torch.sqrt(var)
        
        return self.weight * x_normalized + self.bias

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class H5SeqDataset(Dataset):
    def __init__(self, csv_files, feature_keys, max_len=None):
        self.df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
        for c in ("feature_dir", "feature_file", "feature_path"):
            if c not in self.df.columns:
                raise ValueError(f"Missing column {c}")
        self.feature_keys = feature_keys
        self.max_len      = max_len
        self.labels       = self.df["label"].fillna(-1).astype(np.float32).values
        self.affinity     = self.df["affinity"].astype(float).values
        self._h5_cache    = {}

    # ---------------- internal ----------------
    def _get_handle(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r", libver="latest")
        return self._h5_cache[path]

    # -------------- public --------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        h5path = os.path.join(row["feature_dir"], row["feature_file"])
        grp    = self._get_handle(h5path)[row["feature_path"]]

        feats = []
        if "s_z" in self.feature_keys:
            arr = grp["s_z"][()]
            feats.append(arr)
        if "pae" in self.feature_keys:
            arr = grp["pae"][()]
            feats.append(arr)
        if "contact" in self.feature_keys:
            arr = grp["contact"][()]
            feats.append(arr)

        feature_matrix = np.concatenate(feats, axis=-1).astype(np.float32)
        if self.max_len and feature_matrix.shape[0] > self.max_len:
            feature_matrix = feature_matrix[:self.max_len]

        bad_mask = ~np.isfinite(feature_matrix)
        if bad_mask.any():
            logging.warning(f"NaN/Inf detected in {h5path}[{grp}]. Replacing with zeros.")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        return {
            "feat": feature_matrix,
            "label": self.labels[idx],
            "affinity": self.affinity[idx],
        }

    def __del__(self):
        for f in self._h5_cache.values():
            try: f.close()
            except Exception: pass
            

# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------
def collate_fn(batch, max_len=None):
    B = len(batch)
    max_len_in_batch = max(b["feat"].shape[0] for b in batch)
    #for b in batch:
    #    print(f'YY0 length_feat:{b["feat"].shape}')
    if max_len:
        max_len_in_batch = min(max_len_in_batch, max_len)
    feat_dim = batch[0]["feat"].shape[1]

    feats = torch.zeros(B, max_len_in_batch, feat_dim, dtype=torch.float32)
    mask  = torch.zeros(B, max_len_in_batch, dtype=torch.bool)
    for i, b in enumerate(batch):
        arr = torch.from_numpy(b["feat"]).float()
        L_i = min(arr.shape[0], max_len_in_batch) if max_len else arr.shape[0]
        feats[i, :L_i] = arr[:L_i]
        mask[i, :L_i]  = True

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    aff_list = [b.get("affinity", np.nan) for b in batch]
    aff = torch.tensor(aff_list, dtype=torch.float32)

    return {
        "feat": feats,
        "mask": mask,
        "label": labels,
        "aff": aff,
    }

# ---------------------------------------------------------------------
# Model with Flash Attention
# ---------------------------------------------------------------------
class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).")
    
    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        #if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        #    print("NaN detected in attention projections!")
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        q = q * self.scaling
        
        q = torch.clamp(q, -15, 15)
        k = torch.clamp(k, -15, 15)
        v = torch.clamp(v, -15, 15)

        # Scaled dot-product attention with FlashAttention optimization
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(bsz, 1, 1, -1)\
                                .expand(-1, self.num_heads, tgt_len, -1)\
                                .reshape(bsz*self.num_heads, tgt_len, -1)
                                
            attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
        else:
            attn_mask = None
            
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        #if torch.isnan(attn_output).any():
        #    print("NaN detected in attention output!")
            
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if torch.isnan(attn_output).any():
            #print("NaN detected after out_proj!")
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
            
        return attn_output

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = FlashAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.norm_ff = StableLayerNorm(dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = StableLayerNorm(d_model)
        self.norm2 = StableLayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

def pool_mask(mask, pool_size):
    B, L = mask.shape
    L_new = L // pool_size
    mask = mask[:, :L_new * pool_size]
    mask = mask.view(B, L_new, pool_size).any(dim=-1)
    return mask

class PairTransformer(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=4, nlayers=4, dropout=0.2, use_flash=True, cnn_channels=[128,256,512], kernel_sizes=[7,5,3], pool_sizes=[4,2,2]):
        super().__init__()
        
        self.cnn_layers = nn.ModuleList()
        self.pool_sizes = pool_sizes
        in_channels = in_dim
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, kernel_sizes, pool_sizes)):
            self.cnn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            self.cnn_layers.append(nn.BatchNorm1d(out_channels))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool1d(pool_size))
            
            in_channels = out_channels
        
        # Calculate the output dimension after CNN
        self.cnn_output_dim = cnn_channels[-1]
        
        # Projection to d_model
        self.input_proj = nn.Linear(self.cnn_output_dim, d_model)
        self.input_norm = StableLayerNorm(d_model)
        self.input_activation = nn.GELU()
        
        if use_flash:
            self.encoder = nn.ModuleList([
                FlashTransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
                for _ in range(nlayers)
            ])
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        
        self.cls_head = nn.Sequential(
            StableLayerNorm(d_model),
            StableLinear(d_model, 1)
        )
        
        self.reg_head = nn.Sequential(
            StableLayerNorm(d_model),
            StableLinear(d_model, 1),
            nn.Sigmoid()
        )
        self.use_flash = use_flash
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            if 'norm' in name and 'weight' in name:
                nn.init.ones_(param)
            if 'norm' in name and 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.modules():
            if isinstance(module, StableLinear):
                module.reset_parameters()

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        
        pool_idx = 0
        for layer in self.cnn_layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool1d):
                mask = pool_mask(mask, self.pool_sizes[pool_idx])
                pool_idx += 1
        
        x = x.transpose(1, 2)
        
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_activation(x)
            
        if self.use_flash:
            # For FlashAttention we need (seq_len, batch, features)
            pad_mask = ~mask
            x = x.transpose(0, 1).contiguous()
            for layer in self.encoder:
                x = layer(x, src_key_padding_mask=pad_mask)
            x = x.transpose(0, 1).contiguous()
        else:
            x = self.encoder(x, src_key_padding_mask=~mask)
        
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=5.0, neginf=-5.0)
        pooled = torch.clamp(pooled, -5.0, 5.0)
        
        cls_output = self.cls_head(pooled).squeeze(1)
        cls_output = torch.nan_to_num(cls_output, nan=0.0, posinf=10.0, neginf=-10.0)
        cls_output = torch.clamp(cls_output, -10.0, 10.0)
        
        reg_output = self.reg_head(pooled).squeeze(1)
            
        #pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return cls_output, reg_output

# ---------------------------------------------------------------------
# Loss / Metrics
# ---------------------------------------------------------------------
def compute_loss(logits, pred_aff, labels, affin, cls_weight=1.0, reg_weight=0.0001):
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_cls = (labels >= 0)
    valid_reg = ~torch.isnan(affin)
    
    if valid_cls.any():
        logits_clamped = torch.clamp(logits[valid_cls], -20, 20)
        
        cls_loss = F.binary_cross_entropy_with_logits(
            logits_clamped, 
            labels[valid_cls]
        )
        total_loss += cls_weight * cls_loss
    
    if valid_reg.any():
        epsilon = 1e-6
        pred_aff_clamped = torch.clamp(pred_aff[valid_reg], epsilon, 1.0 - epsilon)
        
        reg_loss = F.smooth_l1_loss(
            pred_aff_clamped,
            affin[valid_reg]
        )
        total_loss += reg_weight * reg_loss
    
    if total_loss == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        return torch.tensor(1e-6, device=logits.device, requires_grad=True)
    
    return total_loss


def calculate_classification_metrics(labels, probs, threshold=0.5):
    if len(labels) == 0:
        return {
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "specificity": float('nan'),
            "f1": float('nan'),
            "ppv": float('nan'),
            "npv": float('nan')
        }
    
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    f1 = f1_score(labels, preds, zero_division=0)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "ppv": ppv,
        "npv": npv
    }


import torch.nn.functional as F
import torch.distributed as dist

def gather_tensor(list_of_tensors, device):
    world_size = dist.get_world_size()
    local_tensor = torch.cat(list_of_tensors, dim=0)
    local_len    = torch.tensor([local_tensor.size(0)],
                                dtype=torch.long,
                                device=device)

    len_list = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(len_list, local_len)
    lens = torch.stack(len_list)
    max_len   = int(lens.max())

    if local_tensor.size(0) < max_len:
        pad_size = max_len - local_tensor.size(0)
        if local_tensor.dim() == 1:
            local_tensor = F.pad(local_tensor, (0, pad_size))
        else:
            local_tensor = F.pad(local_tensor,
                                 (0, 0, 0, pad_size))

    gather_list = [torch.empty_like(local_tensor)
                   for _ in range(world_size)]
    dist.all_gather(gather_list, local_tensor)

    out = torch.cat([t[:l.item()] for t, l in zip(gather_list, lens)], dim=0)
    return out

def evaluate(model, loader, device, threshold=0.5, args=None, desc="Processing Evaluating"):
    model.eval()
    logits_all, labels_all, aff_pred_all, aff_all = [], [], [], []
    
    if args and args.use_fsdp:
        rank = dist.get_rank()
    else:
        rank = 0
        
    progress_bar = None
    if rank == 0:
        progress_bar = tqdm(total=len(loader), desc=desc, leave=False)
    
    
    with torch.no_grad():
        for batch in loader:
            batch_feat = batch["feat"].to(device, non_blocking=True)
            batch_mask = batch["mask"].to(device, non_blocking=True)
            
            with autocast(enabled=args.use_amp if args else False):
                logits, aff_pred = model(batch_feat, batch_mask)
            
            #logits_all.append(logits)
            #labels_all.append(batch["label"].to(device))
            #aff_pred_all.append(aff_pred)
            #aff_all.append(batch["aff"].to(device))
            logits_all.append(logits.detach().contiguous())
            aff_pred_all.append(aff_pred.detach().contiguous())
            labels_all.append(batch["label"].to(device).detach().contiguous())
            aff_all.append(batch["aff"].to(device).detach().contiguous())
            
            if progress_bar is not None:
                progress_bar.update(1)
                
    if progress_bar is not None:
        progress_bar.close()
    
    if args and args.use_fsdp:
        world_size = dist.get_world_size()
        '''
        logits   = gather_tensor(logits_all,   device).cpu().numpy()
        labels   = gather_tensor(labels_all,   device).cpu().numpy()
        aff_pred = gather_tensor(aff_pred_all, device).cpu().numpy()
        aff      = gather_tensor(aff_all,      device).cpu().numpy()
        
        '''
        if len(logits_all) > 0:
            logits_local = torch.cat(logits_all, dim=0).cpu()
        else:
            logits_local = None
        logits_list = [None] * world_size
        dist.all_gather_object(logits_list, logits_local)
        logits_list = [x for x in logits_list if x is not None]
        logits = torch.cat(logits_list, dim=0).numpy() if logits_list else np.array([])

        if len(labels_all) > 0:
            labels_local = torch.cat(labels_all, dim=0).cpu()
        else:
            labels_local = None
        labels_list = [None] * world_size
        dist.all_gather_object(labels_list, labels_local)
        labels_list = [x for x in labels_list if x is not None]
        labels = torch.cat(labels_list, dim=0).numpy() if labels_list else np.array([])

        if len(aff_pred_all) > 0:
            aff_pred_local = torch.cat(aff_pred_all, dim=0).cpu()
        else:
            aff_pred_local = None
        aff_pred_list = [None] * world_size
        dist.all_gather_object(aff_pred_list, aff_pred_local)
        aff_pred_list = [x for x in aff_pred_list if x is not None]
        aff_pred = torch.cat(aff_pred_list, dim=0).numpy() if aff_pred_list else np.array([])

        if len(aff_all) > 0:
            aff_local = torch.cat(aff_all, dim=0).cpu()
        else:
            aff_local = None
        aff_list = [None] * world_size
        dist.all_gather_object(aff_list, aff_local)
        aff_list = [x for x in aff_list if x is not None]
        aff = torch.cat(aff_list, dim=0).numpy() if aff_list else np.array([])
        
         
    else:
        logits = torch.cat(logits_all).cpu().numpy() if logits_all else np.array([])
        labels = torch.cat(labels_all).cpu().numpy() if labels_all else np.array([])
        aff_pred = torch.cat(aff_pred_all).cpu().numpy() if aff_pred_all else np.array([])
        aff = torch.cat(aff_all).cpu().numpy() if aff_all else np.array([])
        
    # Filter valid samples
    valid_cls = labels >= 0
    valid_reg = ~np.isnan(aff)
    
    # Classification metrics
    cls_metrics = {}
    if np.any(valid_cls):
        probs = 1 / (1 + np.exp(-logits[valid_cls]))
        cls_labels = labels[valid_cls]
        if len(np.unique(cls_labels)) < 2:
            cls_metrics = {k: float('nan') for k in ["accuracy", "precision", "recall", "specificity", "f1", "ppv", "npv", "auc"]}
        else:
            cls_metrics = calculate_classification_metrics(cls_labels, probs, threshold)
            try:
                cls_metrics["auc"] = roc_auc_score(cls_labels, probs)
            except ValueError:
                cls_metrics["auc"] = float('nan')
    else:
        cls_metrics = {k: float('nan') for k in ["accuracy", "precision", "recall", "specificity", "f1", "ppv", "npv", "auc"]}
    
    # Regression metrics
    reg_metrics = {}
    if np.any(valid_reg):
        valid_aff = aff[valid_reg]
        valid_aff_pred = aff_pred[valid_reg]
        finite_mask = np.isfinite(valid_aff) & np.isfinite(valid_aff_pred)

        
        if len(valid_aff) < 2:
            reg_metrics = {
                "mse": float('nan'),
                "rmse": float('nan'),
                "r2": float('nan'),
                "pearson": float('nan')
            }
        else:
            valid_aff = valid_aff[finite_mask]
            valid_aff_pred = valid_aff_pred[finite_mask]
            if len(valid_aff) > 0:
                mse_val = mean_squared_error(valid_aff, valid_aff_pred)
                reg_metrics = {
                    "mse": mse_val,
                    "rmse": np.sqrt(mse_val),
                    "r2": r2_score(valid_aff, valid_aff_pred),
                    "pearson": pearsonr(valid_aff, valid_aff_pred)[0]
                }
            else:
                reg_metrics = {"mse": float('nan'), "rmse": float('nan'), "r2": float('nan'), "pearson": float('nan')}
    else:
        reg_metrics = {"mse": float('nan'), "rmse": float('nan'), "r2": float('nan'), "pearson": float('nan')}
    
    return cls_metrics, reg_metrics

# ---------------------------------------------------------------------
# Fold helper
# ---------------------------------------------------------------------
def gather_csvs(root_dir, fold_idx):
    all_folds = sorted([d for d in os.listdir(root_dir) if d.startswith("fold")])
    val_fold_name = f"fold{fold_idx}"
    train_csvs = []
    val_csvs = []
    for fold in all_folds:
        csvs = glob.glob(os.path.join(root_dir, fold, "*.csv"))
        if fold == val_fold_name:
            val_csvs.extend(csvs)
        else:
            train_csvs.extend(csvs)
    if not train_csvs or not val_csvs:
        raise RuntimeError(f"No csvs found for fold {fold_idx}")
    return train_csvs, val_csvs

# ---------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, delta=0.00005):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        if current_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        return self.early_stop

def train_fold(fold, args, device):
    if args.use_fsdp:
        rank = dist.get_rank()
    else:
        rank = 0
    
    # Setup logging for this fold
    logger = setup_logging(args.save_dir, fold, rank)
    logger.info(f"========== Starting Fold {fold} ==========")
    
    summary_row = None
    if rank == 0:
        metrics_dir = os.path.join(args.save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"fold{fold}_metrics.csv")
        
        with open(metrics_file, "w", newline='') as f:
            writer = csv.writer(f)
            #header = [
            #    "epoch", "best_score", "train_loss", "train_auc", "train_acc", "train_prec", "train_rec",
            #    "train_spec", "train_f1", "train_ppv", "train_npv", "train_mse", "train_rmse", "train_r2", "train_pearson",
            #    "val_auc", "val_acc", "val_prec", "val_rec", "val_spec", "val_f1", "val_ppv", "val_npv",
            #    "val_mse", "val_rmse", "val_r2", "val_pearson", "best_epoch"
            #]
            header = [
                "epoch", "best_score", "train_loss", 
                "val_auc", "val_acc", "val_prec", "val_rec", "val_spec", "val_f1", "val_ppv", "val_npv",
                "val_mse", "val_rmse", "val_r2", "val_pearson", "best_epoch"
            ]
            writer.writerow(header)
    
    train_csvs, val_csvs = gather_csvs(args.root, fold)
    logger.info(f"Found {len(train_csvs)} training CSV files and {len(val_csvs)} validation CSV files")
    print(f'val files: {val_csvs}')
    
    # Initialize datasets and loaders
    #train_ds = ESMFullMatrixDataset(train_csvs, args.feature_keys, args.max_len)
    #val_ds = ESMFullMatrixDataset(val_csvs, args.feature_keys, args.max_len)

    train_ds = H5SeqDataset(train_csvs, args.feature_keys, args.max_len)
    val_ds = H5SeqDataset(val_csvs, args.feature_keys, args.max_len)
    
    if rank == 0:
        logger.info(f"Training CSV files: {len(train_csvs)}")
        logger.info(f"Validation CSV files: {len(val_csvs)}")
        
    collate = lambda b: collate_fn(b, args.max_len)
    
    if args.use_fsdp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True, seed=args.seed
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers, 
        collate_fn=collate, 
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers, 
        collate_fn=collate, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model setup
    sample_dim = train_ds[0]["feat"].shape[1]
    model = PairTransformer(
        sample_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        use_flash=args.use_flash,
        cnn_channels=args.cnn_channels,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes
    ).to(device)
    
    # FSDP wrapping if enabled
    if args.use_fsdp and FSDP_AVAILABLE:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, 
            min_num_params=20000000
        )
        if args.use_amp:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16 if args.use_amp else torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            mixed_precision = None
            
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True
        )
    elif torch.cuda.device_count() > 1 and not args.use_fsdp:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * args.epochs, eta_min=1e-6)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            anneal_strategy='cos',
            pct_start=0.0,
            div_factor=1.0,
            final_div_factor=1e4,
            last_epoch=-1
        )
    scaler = GradScaler(enabled=args.use_amp)
    early_stopping = EarlyStopping(patience=args.patience)
    logger.info(f"Optimizer initial lr: {optimizer.param_groups[0]['lr']}")
    
    if args.use_fsdp:
        stop_signal = torch.tensor(0, device=device)
    else:
        stop_signal = None
    
    # Training state
    #best_val_loss = float('inf')
    best_score = -float('inf')
    best_epoch = -1
    best_model_path = os.path.join(args.save_dir, f"full_fold{fold}_best.pt")
    final_model_path = os.path.join(args.save_dir, f"full_fold{fold}_final.pt")
    
    best_val_metrics = None
    
    if rank == 0:
        epoch_bar = tqdm(total=args.epochs, desc=f"Processing Epoch of Fold {fold}", position=0)
    
    # Training loop
    for epoch in range(args.epochs):
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f'YY current_lr:{current_lr}')
        epoch_start = time.time()
        
        if args.use_fsdp and stop_signal.item() == 1:
            logger.info(f"Early stopping triggered, skipping epoch {epoch+1}")
            break
        
        if args.use_fsdp:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        use_tqdm = (rank == 0) and (not args.use_fsdp or dist.get_rank() == 0)
        batch_bar = None
    
        if use_tqdm:
            batch_bar = tqdm(total=len(train_loader), desc="Processing Batch of Training", leave=False, position=1)
            
        #for batch in train_loader:
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            with autocast(enabled=args.use_amp):
                batch_feat = batch["feat"].to(device, non_blocking=True)
                batch_mask = batch["mask"].to(device, non_blocking=True)
                batch_label = batch["label"].to(device, non_blocking=True)
                batch_aff = batch["aff"].to(device, non_blocking=True)
                
                logits, pred_aff = model(batch_feat, batch_mask)
                
                if not torch.all(torch.isfinite(logits)):
                    logger.warning(f"YY: NaN/Inf logits predicted, skip batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                loss = compute_loss(logits, pred_aff, batch_label, batch_aff)
                #print(f'YY: loss is {loss.item()}, logits={logits}, pred_aff={pred_aff},batch_label={batch_label}, batch_aff={batch_aff}')
                
            # Backpropagation with AMP and FSDP
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as e:
                    logger.error(f"Gradient clipping failed: {str(e)}")
                    optimizer.zero_grad()
                    continue
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()

            scheduler.step()
                          
            current_lr = scheduler.get_last_lr()[0]
            train_loss += loss.item()
            batch_count += 1
            
            if rank == 0 and batch_bar is not None:
                with threading.Lock():
                    batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}", refresh=False)
                    batch_bar.update(1)
        
        avg_train_loss = train_loss / batch_count
        
        if rank == 0 and batch_bar is not None:
            batch_bar.close()
            
        if args.use_fsdp:
            avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = avg_train_loss_tensor.item() / dist.get_world_size()
        
        # Evaluation
        #train_cls, train_reg = evaluate(model, train_loader, device, args.threshold, args)
        val_cls, val_reg = evaluate(model, val_loader, device, args.threshold, args)
        
        #scheduler.step()
        
        if rank == 0:
            # Check for best model (using validation RMSE as primary metric)
            #current_val_loss = val_reg.get("rmse", float('inf'))
            val_auc = val_cls.get('auc', 0)
            val_rmse = val_reg.get('rmse', 10)
            rmse_norm = max(0, 1 - (val_rmse / 10))
            current_score = val_auc + 0*rmse_norm
            
            #if not np.isnan(current_val_loss) and current_val_loss < best_val_loss:
            if not np.isnan(current_score) and current_score > best_score:
                #best_val_loss = current_val_loss
                best_score = current_score
                best_epoch = epoch
                
                if args.use_fsdp and FSDP_AVAILABLE:
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                        torch.save(model.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                
                best_val_metrics = {
                    'epoch': epoch + 1,
                    'best_score': best_score,
                    'auc': val_cls.get('auc', float('nan')),
                    'accuracy': val_cls.get('accuracy', float('nan')),
                    'sensitivity': val_cls.get('recall', float('nan')),
                    'specificity': val_cls.get('specificity', float('nan')),
                    'ppv': val_cls.get('precision', float('nan')),
                    'npv': val_cls.get('npv', float('nan')),
                    'rmse': val_reg.get("rmse", float('inf')),
                    'pearson': val_reg.get('pearson', float('nan')),
                    'model_path': best_model_path
                }
                
                logger.info(f"Saved new best model at epoch {epoch+1} with val best score: {best_score:.4f}")
            
            '''
            if early_stopping(current_score):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                if args.use_fsdp:
                    #stop_signal = torch.tensor(1, device=device)
                    stop_signal.fill_(1)
                else:
                    break
            '''
                
        if args.use_fsdp:
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                logger.info(f"Rank {rank} received early stopping signal")
                break
        #elif rank == 0 and early_stopping(current_score):
        #    break
        
        if rank == 0:
            # Log metrics
            logger.info(f"\nFold {fold}| Epoch {epoch+1}/{args.epochs} | Time: {time.time()-epoch_start:.1f}s | Best_Epoch: {best_epoch+1} | Current learning rate: {current_lr:.7f} | " +
                        f"Train Loss: {avg_train_loss:.4f} | " +
                        f"Val bestscore: {best_score:.4f} | " +
                        f"Val currentscore: {current_score:.4f} | " +
                        f"Val AUC: {val_cls.get('auc', 'nan'):.4f} | " +
                        f"Val Acc: {val_cls.get('accuracy', 'nan'):.4f} | " +
                        f"Val Prec: {val_cls.get('precision', 'nan'):.4f} | " +
                        f"Val Rec: {val_cls.get('recall', 'nan'):.4f} | " +
                        f"Val Spec: {val_cls.get('specificity', 'nan'):.4f} | " +
                        f"Val RMSE: {val_reg.get('rmse', 'nan'):.4f} | " +
                        f"Val Pearson: {val_reg.get('pearson', 'nan'):.4f}")
            
            # Save metrics to CSV
            with open(metrics_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, best_score, avg_train_loss,
                    #train_cls.get('auc', ''),
                    #train_cls.get('accuracy', ''),
                    #train_cls.get('precision', ''),
                    #train_cls.get('recall', ''),
                    #train_cls.get('specificity', ''),
                    #train_cls.get('f1', ''),
                    #train_cls.get('ppv', ''),
                    #train_cls.get('npv', ''),
                    #train_reg.get('mse', ''),
                    #train_reg.get('rmse', ''),
                    #train_reg.get('r2', ''),
                    #train_reg.get('pearson', ''),
                    val_cls.get('auc', ''),
                    val_cls.get('accuracy', ''),
                    val_cls.get('precision', ''),
                    val_cls.get('recall', ''),
                    val_cls.get('specificity', ''),
                    val_cls.get('f1', ''),
                    val_cls.get('ppv', ''),
                    val_cls.get('npv', ''),
                    val_reg.get('mse', ''),
                    val_reg.get('rmse', ''),
                    val_reg.get('r2', ''),
                    val_reg.get('pearson', ''),
                    best_epoch+1 if best_epoch != -1 else ''
                ])
        
        if rank == 0 and epoch_bar is not None:
            epoch_bar.set_postfix(
                train_loss=avg_train_loss,
                val_auc=val_cls.get('auc', float('nan')),
                val_rmse=val_reg.get('rmse', float('nan')), refresh=False
            )
            epoch_bar.update(1)
            
    if rank == 0 and epoch_bar is not None:
        epoch_bar.close()
    
    # Save final model (only on rank 0)
    if rank == 0:
        if args.use_fsdp and FSDP_AVAILABLE:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                torch.save(model.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
    
    if rank == 0:
        logger.info(f"Finished Fold {fold}. Best model at epoch {best_epoch+1} with val auc+rmse score: {best_score:.4f}")
    else:
        logger.info(f"Rank {rank} completed training for Fold {fold}")
    
    if rank == 0:
        if best_val_metrics is None:
            best_val_metrics = {
                'epoch': -1,
                'best_score': float('nan'),
                'auc': float('nan'),
                'accuracy': float('nan'),
                'sensitivity': float('nan'),
                'specificity': float('nan'),
                'ppv': float('nan'),
                'npv': float('nan'),
                'rmse': float('nan'),
                'pearson': float('nan'),
                'model_path': final_model_path
            }
            
        return best_val_metrics
    
    return None

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MHC-peptide binding prediction with Transformer")
    parser.add_argument("--root", required=True, help="Root directory containing fold subdirectories")
    parser.add_argument("--feature_keys", nargs='+', default=["s_z", "pae", "contact"], help="Features to use")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=2000, help="Max sequence length (positions)")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--nlayers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--use_flash", action="store_true", help="Use FlashAttention")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--use_fsdp", action="store_true", help="Use Fully Sharded Data Parallel")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--lr_step_size", type=int, default=5, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Gamma for learning rate scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--cnn_channels", nargs='+', type=int, default=[128, 256, 512], help="CNN channel sizes")
    parser.add_argument("--kernel_sizes", nargs='+', type=int, default=[7, 5, 3], help="CNN kernel sizes")
    parser.add_argument("--pool_sizes", nargs='+', type=int, default=[4, 2, 2], help="CNN pool sizes")
    args = parser.parse_args()

    if args.use_fsdp:
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            timeout=timedelta(hours=1)
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    seed_everything(args.seed)
    
    main_logger = setup_logging(args.save_dir, "main", rank)
    main_logger.info(f"Starting 5-fold cross-validation with device: {device}")
    main_logger.info(f"Using features: {args.feature_keys}")
    main_logger.info(f"Max sequence length: {args.max_len}")
    main_logger.info(f"Using FlashAttention: {args.use_flash}")
    main_logger.info(f"Using AMP: {args.use_amp}")
    main_logger.info(f"Using FSDP: {args.use_fsdp}")
    
    if args.use_fsdp and not FSDP_AVAILABLE:
        main_logger.warning("FSDP requested but not available. Falling back to single GPU training.")
        args.use_fsdp = False
    
    summary_file = os.path.join(args.save_dir, "cv_results_summary.csv")
    if rank == 0:
        with open(summary_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold", "best_epoch", "best_score", "auc", "accuracy", 
                "sensitivity", "specificity", "ppv", "npv", 
                "rmse", "pearson", "model_path"
            ])
    
    if rank == 0:
        fold_bar = tqdm(total=5, desc="Processing Fold", position=0)
    
    # Run cross-validation
    fold_performance = {}
    for fold in range(5, 6):
        fold_metrics = train_fold(fold, args, device)

        if rank == 0:
            if fold_metrics:
                fold_bar.set_postfix(
                    best_score=fold_metrics['best_score'],
                    auc=fold_metrics['auc'],
                    rmse=fold_metrics['rmse'], refresh=False
                )
            fold_bar.update(1)

        if rank == 0 and fold_metrics:
            fold_performance[f"fold{fold}"] = fold_metrics
            
            with open(summary_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    fold,
                    fold_metrics['epoch'],
                    fold_metrics['best_score'],
                    fold_metrics['auc'],
                    fold_metrics['accuracy'],
                    fold_metrics['sensitivity'],
                    fold_metrics['specificity'],
                    fold_metrics['ppv'],
                    fold_metrics['npv'],
                    fold_metrics['rmse'],
                    fold_metrics['pearson'],
                    fold_metrics['model_path']
                ])
            
            main_logger.info(f"Added fold {fold} results to summary table")
    
    if rank == 0:
        fold_bar.close()
        
    if rank == 0 and fold_performance:
        avg_metrics = {}
        for metric in ['best_score', 'auc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'rmse', 'pearson']:
            values = [fold_performance[f"fold{fold}"][metric] for fold in range(5, 6)]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                avg_metrics[metric] = sum(valid_values) / len(valid_values)
            else:
                avg_metrics[metric] = float('nan')
        
        with open(summary_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "average",
                "",
                avg_metrics['best_score'],
                avg_metrics['auc'],
                avg_metrics['accuracy'],
                avg_metrics['sensitivity'],
                avg_metrics['specificity'],
                avg_metrics['ppv'],
                avg_metrics['npv'],
                avg_metrics['rmse'],
                avg_metrics['pearson'],
                ""
            ])
        
        main_logger.info("Cross-validation completed. Performance summary:")
        for fold in range(5, 6):
            metrics = fold_performance[f"fold{fold}"]
            main_logger.info(f"Fold {fold}: "
                            f"Epoch={metrics['epoch']}, "
                            f"BEST_score={metrics['best_score']:.4f}, "
                            f"RMSE={metrics['rmse']:.4f}, "
                            f"AUC={metrics['auc']:.4f}, "
                            f"Accuracy={metrics['accuracy']:.4f}")
        
        main_logger.info(f"Average RMSE: {avg_metrics['rmse']:.4f}, "
                        f"Average AUC: {avg_metrics['auc']:.4f}")
    
    if args.use_fsdp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()