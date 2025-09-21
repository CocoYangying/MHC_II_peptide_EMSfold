#!/usr/bin/env python
"""
extract_inter_features.py
-------------------------
Extracts interface features (s_z, pae, contact) from HDF5 files and 
saves aggregated features to a new CSV file with precomputed feature vectors.
"""
import argparse, os, glob, logging, time
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from datetime import datetime

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)
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

def process_csv(csv_path, feature_keys, output_dir, logger):
    """Process a single CSV file, extract features and save new CSV"""
    # Read original CSV
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    for col in ("feature_dir", "feature_file", "feature_path", "label", "affinity"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Feature scaling factors
    feature_scaling = {
        "s_z": 30,
        "pae": 31.75,
        "contact": 1.0
    }
    
    # Extract features for each row
    feature_vectors = []
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        h5_path = os.path.join(row["feature_dir"], row["feature_file"])
        sample_grp = row["feature_path"]
        sequence = row['sequence']
        
        try:
            with h5py.File(h5_path, "r", libver="latest", swmr=True) as h5_file:
                grp = h5_file[sample_grp]
                feats = []
                
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
                                
                # Extract and scale features
                if "s_z" in feature_keys:
                    arr = features["s_z"]   # [M,P,d_z]
                    arr = arr / feature_scaling["s_z"]
                    feats.append(arr.mean(axis=(0, 1)).astype(np.float32))
                
                if "pae" in feature_keys:
                    arr = features["pae"]   # [M,P]
                    arr = arr / feature_scaling["pae"]
                    feats.append(np.array([arr.mean()], dtype=np.float32))
                
                if "contact" in feature_keys:
                    arr = features["contact"]   # [M,P]
                    arr = arr / feature_scaling["contact"]
                    feats.append(np.array([arr.mean()], dtype=np.float32))
                
                # Combine features into a single vector
                feature_vec = np.concatenate(feats).astype(np.float32)
                
                # Handle NaN/Inf values
                if np.isnan(feature_vec).any() or np.isinf(feature_vec).any():
                    logger.warning(f"NaN/Inf detected in {h5_path}[{sample_grp}]")
                    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
                feature_vectors.append(feature_vec)
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {h5_path}[{sample_grp}]: {str(e)}")
            # Create zero vector of appropriate length
            feat_dim = 0
            if "s_z" in feature_keys:
                feat_dim += 128  # Default s_z dimension
            if "pae" in feature_keys:
                feat_dim += 1
            if "contact" in feature_keys:
                feat_dim += 1
            feature_vectors.append(np.zeros(feat_dim, dtype=np.float32))
    
    # Add feature vectors to DataFrame
    df["feature_vector"] = [vec.tolist() for vec in feature_vectors]
    
    # Save processed CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed2_{os.path.basename(csv_path)}")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed CSV with {len(df)} rows to {output_path}")
    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Extract interface features from HDF5 files")
    parser.add_argument("--input", required=True, help="Input directory containing CSV files")
    parser.add_argument("--output", required=True, help="Output directory for processed CSVs")
    parser.add_argument("--features", default="s_z,pae,contact", help="Comma-separated list of features to extract")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting feature extraction")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Features: {args.features}")
    
    feature_keys = [k.strip() for k in args.features.split(',') 
                   if k.strip() in {"s_z", "pae", "contact"}]
    
    # Find all CSV files in input directory
    #csv_files = glob.glob(os.path.join(args.input, "**", "*esm.csv"), recursive=True)
    all_candidates = glob.glob(os.path.join(args.input, "*esm.csv"))
    csv_files = [file for file in all_candidates if not os.path.basename(file).startswith('processed')]
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    logger.info(f"{csv_files}")
    
    
    
    # Process each CSV file
    processed_files = []
    for csv_path in csv_files:
        try:
            processed_path = process_csv(csv_path, feature_keys, args.output, logger)
            processed_files.append(processed_path)
        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {str(e)}")
    
    logger.info(f"Feature extraction complete. Processed {len(processed_files)} files")

if __name__ == "__main__":
    main()