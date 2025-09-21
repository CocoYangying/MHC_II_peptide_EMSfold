#!/usr/bin/env python
# infer_eval.py

import os
import math
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve,
    mean_squared_error
)
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# import Full-Matrix Transformer
# -----------------------------------------------------------------------------
from cv_fullmat_transformer_full_bal import (
    #ESMFullMatrixDataset,
    H5SeqDataset,
    PairTransformer,
    collate_fn as collate_full
)

# -----------------------------------------------------------------------------
# import Inter-Feature MLP
# -----------------------------------------------------------------------------
from cv_inter_transformer import (
    PrecomputedDataset,
    InterMLP,
    ResidualBlock,
    StableLinear,
    StableLayerNorm,
    collate_fn as collate_inter
)


def plot_confusion_matrix(cm, classes, out_path):
    # Normalize the confusion matrix to percentages (row-wise)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums * 100
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            annot[i,j] = f"{cm[i,j]}\n{cm_normalized[i,j]:.1f}%"
    
    plt.figure(figsize=(5,4))
    # Use normalized values for coloring
    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=100)  # Set fixed range for consistent coloring
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def calculate_youden_threshold(y_true, y_score):
    """Calculate optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, j_scores[optimal_idx], fpr[optimal_idx], tpr[optimal_idx]


def infer_and_evaluate(
    mode: str,
    feature: List[str],
    test_csv: List[str],
    checkpoint: str,
    batch_size: int,
    device: torch.device,
    out_dir: str,
    threshold: float = None,
):
    os.makedirs(out_dir, exist_ok=True)
    dfs = []
    for csv_file in test_csv:
        df = pd.read_csv(csv_file)
        if 'mhc_allele' not in df.columns:
            df['mhc_allele'] = 'ALL'
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    n_samples = len(df)
    pred_scores = np.full(n_samples, np.nan, dtype=float)
    pred_labels = np.full(n_samples, np.nan, dtype=float)
    pred_affs = np.full(n_samples, np.nan, dtype=float)

    if mode == "fullmat":
        ds = H5SeqDataset(
            test_csv,
            feature_keys = feature,
            max_len=15000
        )
        sample_item = ds[0]
        feature_dims = {}
        for key, value in sample_item["features"].items():
            feature_dims[key] = value.shape[1] if len(value.shape) > 1 else 1
            
        collate_fn = collate_full
        model = PairTransformer(
            feature_dims=feature_dims,
            d_model=64,
            nhead=4,
            nlayers=2,
            dropout=0.3,
            use_flash=True,
            cnn_channels=[64, 128],
            kernel_sizes=[5, 3],
            pool_sizes=[2, 2]
        ).to(device)
    else:
        ds = PrecomputedDataset(test_csv)
        collate_fn = collate_inter
        model = InterMLP(
            in_dim = ds[0]["feat"].shape[0],
            hidden_sizes=[512,512,256,256,128],
            dropout=0.2,
            activation="gelu",
            use_residual=True
        ).to(device)

    ck = torch.load(checkpoint, map_location=device)
    sd = ck.get("model", ck)

    new_sd = {}
    for k, v in sd.items():
        new_key = k
        prefixes = ["module.", "_fsdp_wrapped_module."]
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_sd[new_key] = v
    
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("=== Loaded state_dict with strict=False ===")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
   
    model.eval()

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    y_true, y_score, y_pred = [], [], []
    aff_true, aff_pred     = [], []
    alleles_list           = []

    idx = 0
    with torch.no_grad():
        for batch in loader:
            if mode == "fullmat":
                features_dict = batch["features"]
                for key in features_dict:
                    features_dict[key] = features_dict[key].to(device)
                mask = batch["mask"].to(device)
                logits, pred_aff = model(features_dict, mask)
            else:
                feat = batch["feat"].to(device)
                logits, pred_aff = model(feat)

            probs     = torch.sigmoid(logits).cpu().numpy()
            pred_aff  = pred_aff.cpu().numpy()
            batch_sz  = probs.shape[0]

            for i in range(batch_sz):
                allele = df.loc[idx, 'mhc']
                alleles_list.append(allele)

                pred_scores[idx] = float(probs[i])
                pred_labels[idx] = int(probs[i] > (threshold if threshold is not None else 0.5))
                pred_affs[idx] = float(pred_aff[i])

                if 'label' in df.columns:
                    lbl = df.loc[idx, 'label']
                    if not math.isnan(lbl):
                        y_true.append(int(lbl))
                        y_score.append(float(probs[i]))
                        y_pred.append(int(probs[i] > (threshold if threshold is not None else 0.5)))

                if 'affinity' in df.columns:
                    at = df.loc[idx, 'affinity']
                    if not math.isnan(at):
                        aff_true.append(float(at))
                        aff_pred.append(float(pred_aff[i]))

                idx += 1

    df['pred_score'] = pred_scores
    df['pred_label'] = pred_labels
    df['pred_affinity'] = pred_affs
    pred_path = os.path.join(out_dir, "predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"=== Saved predictions to {pred_path} ===")

    row_all = {"allele": "overall"}
    if y_true:
        youden_threshold, youden_j, youden_fpr, youden_tpr = calculate_youden_threshold(y_true, y_score)
        print(f"=== Youden's J optimal threshold: {youden_threshold:.4f} (J={youden_j:.4f}) ===")
        print(f"=== At this threshold: FPR={youden_fpr:.4f}, TPR={youden_tpr:.4f} ===")
        
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        
        tn, fp, fn, tp = cm.ravel()
        print(f"=== Confusion Matrix Details ===")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        
        plot_confusion_matrix(
            cm,
            ['non-binder','binder'],
            os.path.join(out_dir, "confusion_matrix_overall.png")
        )

        auc  = roc_auc_score(y_true, y_score)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        print("=== Overall Classification ===")
        print(f"AUC={auc:.3f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        row_all.update({
                "AUC": auc,
                "Acc": acc,
                "Prec": prec,
                "Rec": rec,
                "F1": f1,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            })

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        
        plt.plot(youden_fpr, youden_tpr, 'ro', markersize=8, 
                 label=f"Youden's J (threshold={youden_threshold:.3f})\nFPR={youden_fpr:.3f}, TPR={youden_tpr:.3f}")
        
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(out_dir, "roc_overall.png"))
        plt.close()

    if aff_true:
        if len(aff_true) >= 2:
            r    = pearsonr(aff_true, aff_pred)[0]
            rmse = math.sqrt(mean_squared_error(aff_true, aff_pred))
            print("\n=== Overall Affinity Regression ===")
            print(f"Pearson r={r:.3f}  RMSE={rmse:.3f}")

            plt.figure()
            plt.scatter(aff_true, aff_pred, alpha=0.6)
            plt.xlabel("True Affinity"); plt.ylabel("Predicted Affinity")
            plt.title(f"Affinity Scatter (r={r:.3f})")
            plt.savefig(os.path.join(out_dir, "affinity_scatter_overall.png"))
            plt.close()
            row_all["Pearson_r"] = r
            row_all["RMSE"]      = rmse
        else:
            row_all["Pearson_r"] = np.nan
            row_all["RMSE"]      = np.nan
            print("\n=== Overall Affinity Regression ===")
            print("Not enough samples to compute Pearson (need >=2).")

    print("\n=== Metrics by MHC Allele ===")
    rows=[]
    for allele in sorted(set(alleles_list)):
        idxs = [i for i,a in enumerate(alleles_list) if a==allele]
        yt = [y_true[i]   for i in idxs if i < len(y_true)]
        ys = [y_score[i]  for i in idxs if i < len(y_score)]
        yp = [y_pred[i]   for i in idxs if i < len(y_pred)]
        at = [aff_true[i] for i in idxs if i < len(aff_true)]
        ap = [aff_pred[i] for i in idxs if i < len(aff_pred)]

        row = {"allele": allele}
        if yt:
            cm = confusion_matrix(yt, yp, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            
            plot_confusion_matrix(
                cm,
                ['non-binder','binder'],
                os.path.join(out_dir, f"confusion_{allele}.png")
            )
            # ROC per-allele
            if len(set(yt)) >= 2:
                p_fpr, p_tpr, _ = roc_curve(yt, ys)
                plt.figure()
                plt.plot(p_fpr, p_tpr, label=f"AUC={roc_auc_score(yt,ys):.3f}")
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC {allele}")
                plt.legend()
                plt.savefig(os.path.join(out_dir, f"roc_{allele}.png"))
                plt.close()
                auc_score = roc_auc_score(yt, ys)
            else:
                auc_score = np.nan

            row.update({
                "AUC": auc_score,
                "Acc": accuracy_score(yt, yp),
                "Prec": precision_score(yt, yp, zero_division=0),
                "Rec": recall_score(yt, yp, zero_division=0),
                "F1": f1_score(yt, yp, zero_division=0),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            })
        else:
            for k in ["AUC","Acc","Prec","Rec","F1","TN","FP","FN","TP"]:
                row[k] = np.nan

        if at:
            if len(at) >= 2:
                r2    = pearsonr(at, ap)[0]
                rmse2 = math.sqrt(mean_squared_error(at, ap))
                # scatter per-allele
                plt.figure()
                plt.scatter(at, ap, alpha=0.6)
                plt.xlabel("True"); plt.ylabel("Pred")
                plt.title(f"{allele} Affinity (r={r2:.3f})")
                plt.savefig(os.path.join(out_dir, f"affinity_{allele}.png"))
                plt.close()

                row["Pearson_r"] = r2
                row["RMSE"]      = rmse2
            else:
                row["Pearson_r"] = np.nan
                row["RMSE"]      = np.nan
        else:
            row["Pearson_r"] = np.nan
            row["RMSE"] = np.nan
        
        rows.append(row)

    rows.append(row_all)
    allele_df = pd.DataFrame(rows).set_index("allele")
    print(allele_df.to_string(float_format="%.3f"))
    allele_df.to_csv(os.path.join(out_dir, "metrics_by_allele.csv"))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Inference & Evaluation for FullMatrix / InterTransformer"
    )
    parser.add_argument("--mode", choices=["fullmat","inter"], required=True)
    parser.add_argument("--feature_keys", nargs='+', default=["s_z", "pae", "contact"], help="Features to use")
    parser.add_argument("--test_csv",   required=True, nargs='+',
                        help="CSV must include columns: sequence, label, affinity, mhc_allele")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Probability threshold for classification (default: 0.5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_and_evaluate(
        mode       = args.mode,
        feature    = args.feature_keys,
        test_csv   = args.test_csv,
        checkpoint = args.checkpoint,
        batch_size = args.batch_size,
        device     = device,
        out_dir    = args.out_dir,
        threshold  = args.threshold,
    )