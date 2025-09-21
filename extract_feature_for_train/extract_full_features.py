#!/usr/bin/env python
"""
preprocess_data.py
==================
Preprocess and merge H5 files for each fold to improve training speed.
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import logging

# Feature scaling factors
FEATURE_SCALING = {
    "s_z": 30,
    "pae": 31.75,
    "contact": 1.0
}

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "preprocess.log")
    
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_non_linker_indices(sequence):
    """
    Calculate indices of non-linker regions based on sequence.
    Returns a list of indices to keep for MHC regions only.
    The last part (peptide) is excluded from the returned indices.
    """
    # Split sequence by colons to identify regions
    parts = sequence.split(':')
    
    # Exclude the last part (peptide)
    mhc_parts = parts[:-1]
    
    # Calculate the start and end indices for each non-linker region in MHC
    indices_to_keep = []
    current_index = 0
    
    for i, part in enumerate(mhc_parts):
        part_length = len(part)
        
        # Add indices for this non-linker region
        indices_to_keep.extend(range(current_index, current_index + part_length))
        
        # Move current_index past this region and the following linker (if any)
        current_index += part_length
        if i < len(mhc_parts) - 1:  # Not the last MHC part
            current_index += 25  # Skip the 25aa linker
    
    return indices_to_keep

def remove_linker_regions(features, non_linker_indices, peptide_length):
    """
    Remove linker regions from features using pre-calculated indices.
    For s_z: remove linker rows only (keep all peptide columns)
    For pae and contact: remove linker rows and columns
    """
    processed_features = {}
    
    for key, data in features.items():
        if key == "s_z":
            # For s_z, remove linker rows but keep all peptide columns
            # s_z shape: (mhc_length + linker_length, peptide_length, 128)
            # We want to remove linker rows, keeping non_linker_indices for rows
            # and all columns (peptide positions)
            processed_features[key] = data[non_linker_indices, :, :]
        
        elif key == "pae" or key == "contact":
            # For pae and contact, remove both rows and columns corresponding to linker
            # pae/contact shape: (mhc_length + linker_length, peptide_length)
            # We want to remove linker rows and keep all peptide columns
            processed_features[key] = data[non_linker_indices, :]
        
        else:
            # For other features, just keep as is
            processed_features[key] = data
    
    return processed_features

def process_sample(h5_path, grp_path, feature_keys, sequence):
    """Process a single sample from H5 file, remove linker regions, then scale and reshape"""
    with h5py.File(h5_path, "r") as f:
        grp = f[grp_path]
        features = {}
        
        # First, get the raw data without scaling or reshaping
        raw_features = {}
        if "s_z" in feature_keys:
            raw_features["s_z"] = grp["s_z"][()]
        
        if "pae" in feature_keys:
            raw_features["pae"] = grp["pae"][()]
        
        if "contact" in feature_keys:
            raw_features["contact"] = grp["contact"][()]
        
        # Calculate non-linker indices for MHC regions only
        non_linker_indices = get_non_linker_indices(sequence)
        
        # Calculate peptide length from sequence
        peptide_part = sequence.split(':')[-1]
        peptide_length = len(peptide_part)
        
        # Remove linker regions
        features = remove_linker_regions(raw_features, non_linker_indices, peptide_length)
        
        # Now apply scaling and reshaping
        if "s_z" in feature_keys:
            features["s_z"] = features["s_z"] / FEATURE_SCALING["s_z"]
            # Reshape to (mhc_length * peptide_length, 128)
            mhc_length = features["s_z"].shape[0]
            features["s_z"] = features["s_z"].reshape(mhc_length * peptide_length, -1)
        
        if "pae" in feature_keys:
            features["pae"] = features["pae"] / FEATURE_SCALING["pae"]
            # Reshape to (mhc_length * peptide_length, 1)
            mhc_length = features["pae"].shape[0]
            features["pae"] = features["pae"].reshape(mhc_length * peptide_length, 1)
        
        if "contact" in feature_keys:
            features["contact"] = features["contact"] / FEATURE_SCALING["contact"]
            # Reshape to (mhc_length * peptide_length, 1)
            mhc_length = features["contact"].shape[0]
            features["contact"] = features["contact"].reshape(mhc_length * peptide_length, 1)
        
        # Check for NaN/Inf values
        for key in features:
            bad_mask = ~np.isfinite(features[key])
            if bad_mask.any():
                logging.warning(f"NaN/Inf detected in {h5_path}[{grp_path}][{key}]. Replacing with zeros.")
                features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        
        return features

def process_fold(fold_idx, args, logger):
    """Process all samples in a fold and create merged H5 file"""
    logger.info(f"Processing fold {fold_idx}")
    
    # Find all CSV files for this fold
    fold_dir = os.path.join(args.root, f"fold{fold_idx}")
    all_candidates = glob.glob(os.path.join(fold_dir, "*.esm.csv"))
    csv_files = [file for file in all_candidates if not os.path.basename(file).startswith('processed')]
    
    if not csv_files:
        logger.warning(f"No CSV files found for fold {fold_idx}")
        return
    
    # Read all CSV files
    df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    
    # Create output directory
    output_dir = os.path.join(args.output_root, f"fold{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create merged H5 file
    merged_h5_path = os.path.join(output_dir, f"fold{fold_idx}_merged.h5")
    merged_csv_path = os.path.join(output_dir, f"fold{fold_idx}_merged.csv")
    
    # Process each sample
    updated_rows = []
    
    with h5py.File(merged_h5_path, "w") as merged_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing fold {fold_idx}"):
            h5_path = os.path.join(row["feature_dir"], row["feature_file"])
            grp_path = row["feature_path"]
            sequence = row["sequence"]  # Get the sequence to identify linker regions
            
            try:
                # Process the sample and remove linker regions
                features = process_sample(h5_path, grp_path, args.feature_keys, sequence)
                
                # Create a unique group name for the sample
                sample_id = f"sample_{idx:06d}"
                
                # Create group in merged H5 file
                sample_grp = merged_f.create_group(sample_id)
                
                # Store each feature separately
                for key, data in features.items():
                    sample_grp.create_dataset(key, data=data, compression="gzip")
                
                # Update row with new path information
                updated_row = row.copy()
                updated_row["feature_dir"] = output_dir
                updated_row["feature_file"] = f"fold{fold_idx}_merged.h5"
                updated_row["feature_path"] = sample_id
                
                updated_rows.append(updated_row)
                
            except Exception as e:
                logger.error(f"Error processing {h5_path}[{grp_path}]: {str(e)}")
    
    # Save updated CSV
    if updated_rows:
        updated_df = pd.DataFrame(updated_rows)
        updated_df.to_csv(merged_csv_path, index=False)
        logger.info(f"Saved merged data for fold {fold_idx}: {len(updated_df)} samples")
    else:
        logger.warning(f"No samples processed for fold {fold_idx}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and merge H5 files for each fold")
    parser.add_argument("--root", required=True, help="Root directory containing fold subdirectories")
    parser.add_argument("--output_root", required=True, help="Output directory for merged files")
    parser.add_argument("--feature_keys", nargs='+', default=["s_z", "pae", "contact"], help="Features to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(args.output_root, "logs")
    logger = setup_logging(log_dir)
    
    logger.info(f"Starting data preprocessing with features: {args.feature_keys}")
    
    # Process each fold
    for fold_idx in range(5, 6):
        process_fold(fold_idx, args, logger)
    
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()