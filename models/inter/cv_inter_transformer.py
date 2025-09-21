#!/usr/bin/env python
"""
train_inter_mlp.py
------------------
Trains MLP model using precomputed interface features from CSV files.

Author: Yang Ying YYANG047@e.ntu.edu.sg
"""
import argparse, os, glob, logging, time, math
import pandas as pd, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score
from scipy.stats import pearsonr
import os
import csv
from datetime import datetime
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from datetime import timedelta
import functools

from tqdm import tqdm

# 条件导入FSDP
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp import MixedPrecision
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_SOCKET_TIMEOUT'] = '3600'
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["NCCL_SOCKET_NTHREADS"] = "4"

# Configure logging
def setup_logging(save_dir, fold, rank):
    """Setup logging with both console and file handlers"""
    log_dir = os.path.join(save_dir, f"fold{fold}_logs")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"inter_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(f"fold{fold}_rank{rank}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler (only rank 0)
    if rank == 0:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        f"Rank {rank} | %(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)
    
    return logger

def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
        
        output = F.linear(x_clamped, weight_clamped, self.bias)
        
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

# Dataset for precomputed features -------------------------------------------------
class PrecomputedDataset(Dataset):
    def __init__(self, csv_files):
        self.df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
        
        # Validate required columns
        required_cols = ["label", "affinity", "feature_vector"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert feature vectors from strings to arrays
        self.features = self.df["feature_vector"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=',', dtype=np.float32)
        ).values
        
        self.labels = self.df["label"].fillna(-1).astype(np.float32).values
        self.affinity = self.df["affinity"].astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "feat":     self.features[idx],
            "label":    self.labels[idx],
            "affinity": self.affinity[idx]
        }

# Collate function -----------------------------------------------------------------
def collate_fn(batch):
    feat_arr = np.stack([b["feat"] for b in batch], axis=0)
    feats = torch.from_numpy(feat_arr).float()
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    aff_list = [b.get("affinity", np.nan) for b in batch]
    aff = torch.tensor(aff_list, dtype=torch.float32)
    return {"feat": feats, "label": labels, "aff": aff}

# Model ----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation="gelu", dropout=0.2):
        super().__init__()
        act_layer = nn.GELU() if activation.lower() == "gelu" else (
            nn.SiLU() if activation.lower() == "silu" else nn.LeakyReLU(0.1)
        )
        self.block = nn.Sequential(
            StableLayerNorm(dim),
            StableLinear(dim, dim),
            act_layer,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class InterMLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=[512, 256, 128], dropout=0.2, activation="gelu", use_residual=True):
        super().__init__()

        layers = [StableLayerNorm(in_dim)]
        prev_dim = in_dim
        act_layer = nn.GELU() if activation.lower() == "gelu" else (
            nn.SiLU() if activation.lower() == "silu" else nn.LeakyReLU(0.1)
        )

        for h in hidden_sizes:
            layers.append(StableLinear(prev_dim, h))
            layers.append(act_layer)
            layers.append(nn.Dropout(dropout))
            layers.append(StableLayerNorm(h))
            if use_residual and prev_dim == h:  # Only add residual when dimensions match
                layers.append(ResidualBlock(h, activation=activation, dropout=dropout))
            prev_dim = h

        self.net = nn.Sequential(*layers)

        # Classification head
        self.cls_head = nn.Sequential(
            StableLayerNorm(prev_dim),
            StableLinear(prev_dim, 1)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            StableLayerNorm(prev_dim),
            StableLinear(prev_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, StableLinear):
                m.reset_parameters()

    def forward(self, x):
        features = self.net(x)
        features = torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = torch.clamp(features, -5.0, 5.0)

        cls_output = self.cls_head(features).squeeze(1)
        cls_output = torch.nan_to_num(cls_output, nan=0.0, posinf=10.0, neginf=-10.0)
        cls_output = torch.clamp(cls_output, -10.0, 10.0)

        reg_output = self.reg_head(features).squeeze(1)
        return cls_output, reg_output

# Loss function --------------------------------------------------------------------
def loss_fn(logits, pred_aff, labels, affin, cls_weight=1.0, reg_weight=0.1):
    terms = []
    total_loss = torch.tensor(0.0, device=logits.device)
    
    if (labels >= 0).any():
        logits_clamped = torch.clamp(logits[labels >= 0], -20, 20)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            logits_clamped,
            labels[labels >= 0],
            reduction='mean'
        )
        terms.append(("cls", ce_loss, cls_weight))
    
    if (~torch.isnan(affin)).any():
        epsilon = 1e-6
        pred_aff_clamped = torch.clamp(pred_aff[~torch.isnan(affin)], epsilon, 1.0-epsilon)
        
        mse_loss = F.smooth_l1_loss(
            pred_aff_clamped,
            affin[~torch.isnan(affin)],
            reduction='mean'
        )
        terms.append(("reg", mse_loss, reg_weight))
    
    for task_type, loss, weight in terms:
        total_loss = total_loss + weight * loss
    
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        return torch.tensor(1e-6, device=logits.device, requires_grad=True)
    
    return total_loss

# Metrics --------------------------------------------------------------------------
def calculate_classification_metrics(labels, probs):
    """Calculate classification metrics from labels and probabilities"""
    if len(labels) == 0:
        return {
            "accuracy": float('nan'),
            "sensitivity": float('nan'),
            "specificity": float('nan'),
            "ppv": float('nan'),
            "npv": float('nan'),
            "auc": float('nan')
        }
    
    # Calculate AUC
    auc = roc_auc_score(labels, probs)
    
    # Calculate confusion matrix metrics
    preds = (probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "auc": auc
    }

def calculate_regression_metrics(true_aff, pred_aff):
    """Calculate regression metrics from true and predicted affinities"""
    if len(true_aff) == 0:
        return {
            "mse": float('nan'),
            "rmse": float('nan'),
            "pearson": float('nan')
        }
    
    mse = mean_squared_error(true_aff, pred_aff)
    rmse = math.sqrt(mse)
    pearson, _ = pearsonr(true_aff, pred_aff)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson
    }

def evaluate(model, loader, device, args=None, desc = "Evaluating"):
    """Evaluate model and return comprehensive metrics (with FSDP support)"""
    model.eval()
    all_logits, all_labels, all_pred_aff, all_true_aff = [], [], [], []
    
    if args and args.use_fsdp:
        rank = dist.get_rank()
    else:
        rank = 0
        
    progress_bar = None
    if rank == 0:
        progress_bar = tqdm(total=len(loader), desc=desc, leave=False)
        
    with torch.no_grad():
        for batch in loader:
            x = batch["feat"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            aff = batch["aff"].to(device, non_blocking=True)
            
            with autocast(enabled=args.use_amp if args else False):
                logits, pred_aff = model(x)
            
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            all_pred_aff.append(pred_aff.detach())
            all_true_aff.append(aff.detach())

            if progress_bar is not None:
                progress_bar.update(1)
                
    if progress_bar is not None:
        progress_bar.close()
        
    
    if args and args.use_fsdp:
        world_size = dist.get_world_size()
        if len(all_logits) > 0:
            logits_local = torch.cat(all_logits, dim=0).cpu()
        else:
            logits_local = None
        logits_list = [None] * world_size
        dist.all_gather_object(logits_list, logits_local)
        logits_list = [x for x in logits_list if x is not None]
        logits = torch.cat(logits_list, dim=0).numpy() if logits_list else np.array([])

        if len(all_labels) > 0:
            labels_local = torch.cat(all_labels, dim=0).cpu()
        else:
            labels_local = None
        labels_list = [None] * world_size
        dist.all_gather_object(labels_list, labels_local)
        labels_list = [x for x in labels_list if x is not None]
        labels = torch.cat(labels_list, dim=0).numpy() if labels_list else np.array([])

        if len(all_pred_aff) > 0:
            aff_pred_local = torch.cat(all_pred_aff, dim=0).cpu()
        else:
            aff_pred_local = None
        aff_pred_list = [None] * world_size
        dist.all_gather_object(aff_pred_list, aff_pred_local)
        aff_pred_list = [x for x in aff_pred_list if x is not None]
        pred_aff = torch.cat(aff_pred_list, dim=0).numpy() if aff_pred_list else np.array([])

        if len(all_true_aff) > 0:
            aff_local = torch.cat(all_true_aff, dim=0).cpu()
        else:
            aff_local = None
        aff_list = [None] * world_size
        dist.all_gather_object(aff_list, aff_local)
        aff_list = [x for x in aff_list if x is not None]
        true_aff = torch.cat(aff_list, dim=0).numpy() if aff_list else np.array([])
    else:
        logits = torch.cat(all_logits).cpu().numpy() if all_logits else np.array([])
        labels = torch.cat(all_labels).cpu().numpy() if all_labels else np.array([])
        pred_aff = torch.cat(all_pred_aff).cpu().numpy() if all_pred_aff else np.array([])
        true_aff = torch.cat(all_true_aff).cpu().numpy() if all_true_aff else np.array([])
        
    cls_mask = (labels >= 0)
    reg_mask = ~np.isnan(true_aff)
    
    cls_labels = labels[cls_mask]
    if np.any(cls_mask):
        cls_probs = 1 / (1 + np.exp(-logits[cls_mask].astype(np.float32)))
    else:
        cls_probs = np.array([])
    
    reg_true_aff = true_aff[reg_mask]
    reg_pred_aff = pred_aff[reg_mask]
    
    cls_metrics = calculate_classification_metrics(cls_labels, cls_probs)
    reg_metrics = calculate_regression_metrics(reg_true_aff, reg_pred_aff)
    
    return {**cls_metrics, **reg_metrics}

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

# Folds -------------------------------------------------------------------
def gather_csvs(root_dir, fold_idx):
    val_fold = f"fold{fold_idx}"
    train_csvs, val_csvs = [], []
    for fold in sorted(os.listdir(root_dir)):
        csvs = glob.glob(os.path.join(root_dir, fold, "processed2*.csv"))
        if fold == val_fold: val_csvs.extend(csvs)
        else: train_csvs.extend(csvs)
    if not train_csvs or not val_csvs:
        raise RuntimeError("CSV files missing for CV")
    return train_csvs, val_csvs

# Main training function --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root dir with fold1..fold5/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", default="checkpoints/inter_cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fsdp", action="store_true", help="Use Fully Sharded Data Parallel")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--hidden", type=str, default="512,512,256,256,128", help="Comma-separated list of hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr_step_size", type=int, default=5, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Gamma for learning rate scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable TQDM progress bars")

    args = parser.parse_args()

    # Initialize distributed training
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
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    args.rank = rank
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    
    # Prepare results storage (only rank 0)
    if rank == 0:
        results_dir = os.path.join(args.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        all_folds_results = []
    
    hidden = [int(x) for x in args.hidden.split(",")]
    
    for fold in range(1, 6):
        logger = setup_logging(args.save_dir, fold, rank)
        logger.info(f"==== Fold {fold} ====")
        logger.info(f"Device: {device} (Rank {rank}/{world_size})")
        logger.info(f"Save directory: {args.save_dir}")
        
        train_csvs, val_csvs = gather_csvs(args.root, fold)
        logger.info(f"Train CSVs: {len(train_csvs)}, Val CSVs: {len(val_csvs)}")

        # Use our new dataset class for precomputed features
        train_ds = PrecomputedDataset(train_csvs)
        val_ds = PrecomputedDataset(val_csvs)
        
        # Create distributed sampler
        if args.use_fsdp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
            
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, 
            sampler=train_sampler,
            num_workers=16, 
            collate_fn=collate_fn, 
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, 
            sampler=val_sampler,
            num_workers=16, 
            collate_fn=collate_fn, 
            pin_memory=True,
            persistent_workers=True
        )

        # Determine input dimension from first sample
        in_dim = train_ds[0]["feat"].shape[0]
        model = InterMLP(in_dim=in_dim, hidden_sizes=hidden, dropout=args.dropout, 
                         activation="gelu", use_residual=True).to(device)

        # FSDP wrapping
        if args.use_fsdp and FSDP_AVAILABLE:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, 
                min_num_params=100000
            )
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float16 if args.use_amp else torch.float32,
                    buffer_dtype=torch.float32,
                ) if args.use_amp else None,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                use_orig_params=True
            )
            logger.info(f"Model wrapped with FSDP")
        elif torch.cuda.device_count() > 1 and not args.use_fsdp:
            model = nn.DataParallel(model)
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        early_stopping = EarlyStopping(patience=args.patience)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

        if args.use_fsdp:
            stop_signal = torch.tensor(0, device=device)
        else:
            stop_signal = None

        # Prepare metrics storage (only rank 0)
        if rank == 0:
            metrics_file = os.path.join(results_dir, f"fold{fold}_metrics.csv")
            with open(metrics_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_auc", 
                                "val_rmse", "val_accuracy", "val_sensitivity", "val_specificity",
                                "val_ppv", "val_npv", "val_pearson", "best_score"])

        best_score = -float('inf')
        best_epoch = -1
        best_metrics = {}
        best_model_path = os.path.join(args.save_dir, f"inter_fold{fold}_best.pt")
        final_model_path = os.path.join(args.save_dir, f"inter_fold{fold}_final.pt")

        if rank == 0 and not args.no_tqdm:
            fold_pbar = tqdm(total=args.epochs, desc=f"Processing Epoch of Fold {fold} Training", position=0)
        else:
            fold_pbar = None

        for epoch in range(args.epochs):
            if args.use_fsdp:
                dist.barrier()
            
            if args.use_fsdp and stop_signal.item() == 1:
                logger.info(f"Early stopping triggered, skipping epoch {epoch+1}")
                break
            
            if args.use_fsdp:
                train_loader.sampler.set_epoch(epoch)
                
            # Training phase
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            if rank == 0 and not args.no_tqdm:
                batch_pbar = tqdm(total=len(train_loader), desc=f"Processing batch of Epoch {epoch+1}", 
                                  leave=False, position=1)
            else:
                batch_pbar = None
            
            for batch_idx, batch in enumerate(train_loader):
                x = batch["feat"].to(device)
                labels = batch["label"].to(device)
                aff = batch["aff"].to(device)

                with autocast(enabled=args.use_amp):
                    logits, affp = model(x)
                    loss = loss_fn(logits, affp, labels, aff)
                
                optimizer.zero_grad()
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_pbar is not None:
                    batch_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"}, refresh=False)
                    batch_pbar.update(1)
                    
            if batch_pbar is not None:
                batch_pbar.close()
                batch_pbar = None

            # Synchronize training loss across processes
            if args.use_fsdp:
                avg_train_loss = torch.tensor(epoch_loss / num_batches, device=device)
                dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
                avg_train_loss = avg_train_loss.item() / world_size
            else:
                avg_train_loss = epoch_loss / num_batches
            
            # Evaluation phase
            val_metrics = evaluate(model, val_loader, device, args)
            
            # Main process handles metrics
            if rank == 0:
                # Calculate composite score
                rmse_norm = max(0, 1 - (val_metrics["rmse"] / 10))
                current_score = val_metrics["auc"] + 0.1*rmse_norm
                
                # Update best model
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_metrics = val_metrics.copy()
                    
                    # Save best model
                    if args.use_fsdp and FSDP_AVAILABLE:
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                            torch.save(model.state_dict(), best_model_path)
                    else:
                        torch.save(model.state_dict(), best_model_path)
                    logger.info(f"\n*** New best model saved (Score: {best_score:.4f}) ***")
                
                # Log metrics
                logger.info(f"\nFold {fold} | Epoch {epoch+1:3d} Summary: | Best_Epoch {best_epoch+1} | current_lr: {current_lr:.7f} | " +
                    f"Train Loss: {avg_train_loss:.4f} | " +
                    f"Val Score: {current_score:.4f} | AUC: {val_metrics['auc']:.4f} | " +
                    f"Acc: {val_metrics['accuracy']:.4f} | RMSE: {val_metrics['rmse']:.4f} | " +
                    f"Pearson: {val_metrics['pearson']:.4f}")
                
                # Save epoch metrics
                with open(metrics_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch+1, avg_train_loss, 
                        val_metrics["auc"], val_metrics["rmse"], val_metrics["accuracy"],
                        val_metrics["sensitivity"], val_metrics["specificity"],
                        val_metrics["ppv"], val_metrics["npv"], val_metrics["pearson"],
                        current_score
                    ])
                
                '''
                if early_stopping(current_score):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    if args.use_fsdp:
                        stop_signal.fill_(1)
                    else:
                        break
                '''
                
                if fold_pbar is not None:
                    fold_pbar.set_postfix({
                        "best_auc": f"{best_metrics.get('auc', 0):.4f}",
                        "best_rmse": f"{best_metrics.get('rmse', 0):.4f}",
                        "epoch": f"{epoch+1}/{args.epochs}"
                    }, refresh=False)
                    fold_pbar.update(1)
                
            if args.use_fsdp:
                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1:
                    logger.info(f"Rank {rank} received early stopping signal")
                    break

            if rank == 0:
                # Save periodic checkpoint
                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    checkpoint_path = os.path.join(args.save_dir, f"inter_fold{fold}_epoch{epoch+1}.pt")
                    
                    if args.use_fsdp and FSDP_AVAILABLE:
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                            torch.save(model.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.state_dict(), checkpoint_path)
                    
                    logger.info(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
            torch.cuda.empty_cache()
            if args.use_fsdp:
                dist.barrier()
        
        if fold_pbar is not None:
            fold_pbar.close()
            fold_pbar = None 
        
        # Main process logs best model
        if rank == 0:
            logger.info(f"\n*** Fold {fold} Best Model (Epoch {best_epoch+1}) | " +
                f"AUC: {best_metrics['auc']:.4f} | Acc: {best_metrics['accuracy']:.4f} | " +
                f"RMSE: {best_metrics['rmse']:.4f} | Pearson: {best_metrics['pearson']:.4f} | " +
                f"Composite Score: {best_score:.4f} | Model: {best_model_path}\n")
            
            # Store results for final summary
            all_folds_results.append({
                "fold": fold,
                "best_epoch": best_epoch+1,
                "best_score": best_score,
                "auc": best_metrics["auc"],
                "accuracy": best_metrics["accuracy"],
                "sensitivity": best_metrics["sensitivity"],
                "specificity": best_metrics["specificity"],
                "ppv": best_metrics["ppv"],
                "npv": best_metrics["npv"],
                "rmse": best_metrics["rmse"],
                "pearson": best_metrics["pearson"],
                "model_path": best_model_path
            })
        
        if rank == 0:
            if args.use_fsdp and FSDP_AVAILABLE:
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    torch.save(model.state_dict(), final_model_path)
            else:
                torch.save(model.state_dict(), final_model_path)  
    
    # Final summary (only rank 0)
    if rank == 0:
        summary_file = os.path.join(results_dir, "cross_validation_summary.csv")
        with open(summary_file, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "fold", "best_epoch", "best_score", "auc", "accuracy", 
                "sensitivity", "specificity", "ppv", "npv", "rmse", "pearson", "model_path"
            ])
            writer.writeheader()
            writer.writerows(all_folds_results)
        
        # Calculate average metrics
        avg_metrics = {
            "auc": np.mean([r["auc"] for r in all_folds_results]),
            "accuracy": np.mean([r["accuracy"] for r in all_folds_results]),
            "sensitivity": np.mean([r["sensitivity"] for r in all_folds_results]),
            "specificity": np.mean([r["specificity"] for r in all_folds_results]),
            "ppv": np.mean([r["ppv"] for r in all_folds_results]),
            "npv": np.mean([r["npv"] for r in all_folds_results]),
            "rmse": np.mean([r["rmse"] for r in all_folds_results]),
            "pearson": np.mean([r["pearson"] for r in all_folds_results])
        }
        
        logger = logging.getLogger()
        logger.info("\n===== CROSS-VALIDATION SUMMARY =====")
        logger.info(f"Average AUC: {avg_metrics['auc']:.4f}")
        logger.info(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")
        logger.info(f"Average Sensitivity: {avg_metrics['sensitivity']:.4f}")
        logger.info(f"Average Specificity: {avg_metrics['specificity']:.4f}")
        logger.info(f"Average PPV: {avg_metrics['ppv']:.4f} | NPV: {avg_metrics['npv']:.4f}")
        logger.info(f"Average RMSE: {avg_metrics['rmse']:.4f} | Pearson: {avg_metrics['pearson']:.4f}")
        logger.info(f"Detailed results saved to: {summary_file}")
  
    # Clean up distributed training
    if args.use_fsdp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()