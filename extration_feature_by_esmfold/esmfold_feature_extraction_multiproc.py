#!/usr/bin/env python3
# esmfold_feature_extraction_multiproc.py
# Yang Ying, YYANG047@e.ntu.edu.sg

import os
import re
import gc
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import esm
import logging
import h5py
from datetime import datetime
from tqdm import tqdm
from scipy.special import softmax
import multiprocessing as mp
import glob
import threading


class ProgressMonitor:
    def __init__(self, total_tasks, update_interval=1):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.lock = threading.Lock()
        self.last_update_time = time.time()
        self.update_interval = update_interval
        self.start_time = time.time()
        self.running = True
        self.progress_thread = threading.Thread(target=self._monitor_progress)
        self.progress_thread.daemon = True
        self.progress_thread.start()
    
    def _monitor_progress(self):
        while self.running:
            with self.lock:
                current_completed = self.completed_tasks
                elapsed = time.time() - self.start_time
                
                if current_completed > 0:
                    items_per_sec = current_completed / elapsed
                    remaining = self.total_tasks - current_completed
                    if items_per_sec > 0:
                        eta_seconds = remaining / items_per_sec
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    else:
                        eta_str = "N/A"
                    
                    percent = (current_completed / self.total_tasks) * 100
                    
                    print(f"\r[Progress] {current_completed}/{self.total_tasks} ({percent:.1f}%) | "
                          f"Speed: {items_per_sec:.2f} seq/sec | ETA: {eta_str}", end="", flush=True)
            
            time.sleep(self.update_interval)
    
    def update(self, increment=1):
        with self.lock:
            self.completed_tasks += increment
    
    def stop(self):
        self.running = False
        self.progress_thread.join()
        elapsed = time.time() - self.start_time
        if self.completed_tasks > 0:
            items_per_sec = self.completed_tasks / elapsed
            print(f"\n[Progress] Completed {self.completed_tasks}/{self.total_tasks} sequences "
                  f"in {elapsed:.2f} seconds ({items_per_sec:.2f} seq/sec)")

def clean_seq(seq: str) -> str:
    s = re.sub("[^A-Z:X]", "", seq.upper())
    s = re.sub(":+", ":", s).strip(":")
    
    return s

def extract_raw_stats(out, idx):
    s_z = out["s_z"][idx]
    pae = out["predicted_aligned_error"][idx]
    dlogits = out["distogram_logits"][idx]
    ci = out["chain_index"][idx]

    n_chains = int(ci.max().item()) + 1
    
    pep_mask = (ci == n_chains - 1)
    mhc_mask = (ci < n_chains - 1)

    if not pep_mask.any():
        logging.warning(f"No peptide chain found in sample {idx}. Chain indices: {ci}, n_chains={n_chains}")
        return None, None, None
    if not mhc_mask.any():
        logging.warning(f"No MHC chain found in sample {idx}. Chain indices: {ci}, n_chains={n_chains}")
        return None, None, None

    mhc_idx = mhc_mask.nonzero().squeeze()
    pep_idx = pep_mask.nonzero().squeeze()

    patch_sz = s_z[mhc_idx][:, pep_idx, :].cpu().numpy()
    patch_pae = pae[mhc_idx][:, pep_idx].cpu().numpy()

    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    probs = softmax(dlogits.cpu().numpy(), axis=-1)
    cmap = probs[..., bins < 8].sum(-1)
    patch_ct = cmap[mhc_mask.cpu().numpy()][:, pep_mask.cpu().numpy()]
    
    
    logging.debug(f"Extracted features for sample {idx}: MHC chains={len(mhc_idx)}, "
                 f"Peptide length={len(pep_idx)}, s_z shape={patch_sz.shape}, "
                 f"pae shape={patch_pae.shape}, contact shape={patch_ct.shape}")
    
    return patch_sz, patch_pae, patch_ct

class H5FeatureWriter:
    def __init__(self, feature_dir, shard_id, group_size=1000):
        self.feature_dir = feature_dir
        self.shard_id = shard_id
        self.group_size = group_size
        self.current_group = 0
        self.counter = 0
        self.file = None
        self.current_file_path = None
        self._find_last_group()
        self._create_new_file()
    
    def _find_last_group(self):
        pattern = os.path.join(
            self.feature_dir, 
            f"features_shard{self.shard_id}_group*.h5"
        )
        existing = glob.glob(pattern)
        if existing:
            groups = []
            for f in existing:
                try:
                    base = os.path.basename(f)
                    group_num = int(base.split("_group")[-1].split('.')[0])
                    groups.append(group_num)
                except:
                    continue
            
            if groups:
                last_group = max(groups)
                self.current_group = last_group + 1
                logging.info(f"Resuming from group {self.current_group} for shard {self.shard_id}")
                return
        self.current_group = 0
    
    def _create_new_file(self):
        if self.file:
            self.file.close()
            logging.info(f"Closed previous feature file")
            self.file = None
        
        file_path = os.path.join(
            self.feature_dir,
            f"features_shard{self.shard_id}_group{self.current_group}.h5"
        )
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self.file = h5py.File(file_path, 'w')
        self.current_file_path = file_path
        self.file.attrs['shard_id'] = self.shard_id
        self.file.attrs['group_id'] = self.current_group
        self.file.attrs['created_at'] = datetime.now().isoformat()
        self.counter = 0
        logging.info(f"Created new feature file: {file_path}")
        return file_path
    
    def add_feature(self, s_z, pae, contact):
        if self.counter >= self.group_size:
            self.current_group += 1
            self._create_new_file()
        
        sample_name = f"sample_{self.counter}"
        grp = self.file.create_group(sample_name)
        
        grp.create_dataset('s_z', data=s_z, compression='gzip', 
                          chunks=(min(32, s_z.shape[0]), min(32, s_z.shape[1]), s_z.shape[2]))
        grp.create_dataset('pae', data=pae, compression='gzip', 
                          chunks=(min(32, pae.shape[0]), min(32, pae.shape[1])))
        grp.create_dataset('contact', data=contact, compression='gzip', 
                          chunks=(min(32, contact.shape[0]), min(32, contact.shape[1])))
        
        self.counter += 1
        return self.current_file_path, sample_name
    
    def close(self):
        if self.file:
            file_path = self.current_file_path
            self.file.close()
            self.file = None
            logging.info(f"Closed feature file: {file_path}")
        else:
            logging.info("No open file to close")

def process_batch(args, device, model, batch):
    results = []
    seqs = []
    valid_rows = []
    
    for i, row in batch.iterrows():
        try:
            cleaned = clean_seq(row['sequence'])
            seqs.append(cleaned)
            valid_rows.append(row)
        except ValueError as e:
            logging.warning(f"Invalid sequence at index {i}: {e}")
    
    if not seqs:
        return results
    
    try:
        with torch.no_grad():
            linker = 'X' * args.linker_len
            out = model.infer(
                seqs,
                num_recycles=args.recycle,
                chain_linker=linker,
                residue_index_offset=512,
            )
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logging.warning(f"Runtime error during inference (OOM): {e}")
            torch.cuda.empty_cache()
            raise
        else:
            logging.warning(f"Runtime error during inference: {e}")
            logging.debug(f"Error sequences: {seqs}")
            return results
    except Exception as e:
        logging.warning(f"Unexpected error during inference: {e}")
        return results
    
    for j, row in enumerate(valid_rows):
        try:
            raw_sz, raw_pae, raw_ct = extract_raw_stats(out, j)
            if raw_sz is not None:
                results.append({
                    'sequence': row['sequence'],
                    'label': row['label'],
                    'affinity': row['affinity'],
                    'mhc': row['mhc'],
                    'peptide': row['peptide'],
                    's_z': raw_sz,
                    'pae': raw_pae,
                    'contact': raw_ct
                })
            else:
                logging.warning(f"Feature extraction returned None for sequence: {seqs[j]}")
        except Exception as e:
            logging.warning(f"Error extracting features for sequence {j}: {e}")
            logging.debug(f"Problem sequence: {seqs[j]}")
    
    return results


def process_shard(shard_id, df_chunk, args, feature_dir, result_queue, progress_queue):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        feature_dir, 
        f"feature_extract.shard{shard_id}.{timestamp}.log"
    )
    skipped_file = os.path.join(feature_dir, f"shard{shard_id}_oom_sequences.csv")
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Shard {shard_id}] %(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start processing shard {shard_id} with {len(df_chunk)} sequences")
    
    if not os.path.exists(skipped_file):
        with open(skipped_file, "w") as f:
            f.write("index,sequence,length,error_type\n")
    
    device_id = shard_id % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    logger.info(f"Using device: {device}")
    
    model = esm.pretrained.esmfold_v1()
    model = model.to(torch.float32)
    model.set_chunk_size(args.chunk_size)
    model.eval()
    
    model = model.to(device)
    
    torch.cuda.empty_cache()
    torch.backends.cuda.max_split_size_mb = 128
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    logger.info(f"Model loaded on {device}")
    
    feature_writer = H5FeatureWriter(feature_dir, shard_id, args.group_size)
    
    df_chunk = df_chunk.copy()
    df_chunk['cleaned_sequence'] = df_chunk['sequence'].apply(
        lambda x: clean_seq(x) if pd.notnull(x) else ""
    )
    df_chunk['length'] = df_chunk['cleaned_sequence'].apply(len)
    df_chunk = df_chunk.sort_values('length', ascending=False)
    
    all_results = []
    skipped_count = 0
    try:
        current_index = 0
        pbar = tqdm(total=len(df_chunk), desc=f"Shard {shard_id}", position=shard_id)
        
        while current_index < len(df_chunk):
            end_index = min(current_index + args.batch_size, len(df_chunk))
            batch = df_chunk.iloc[current_index:end_index]
            batch_oom = False
            
            try:
                batch_results = process_batch(args, device, model, batch)
                
                for result in batch_results:
                    file_path, feature_path = feature_writer.add_feature(
                        result['s_z'], result['pae'], result['contact']
                    )
                    all_results.append({
                        'sequence': result['sequence'],
                        'label': result['label'],
                        'affinity': result['affinity'],
                        'mhc': result['mhc'],
                        'peptide': result['peptide'],
                        'feature_file': os.path.basename(file_path),
                        'feature_path': feature_path
                    })
                
                processed_count = len(batch)
                pbar.update(processed_count)
                progress_queue.put(processed_count)
                
                torch.cuda.empty_cache()
                gc.collect()
                
                current_index = end_index

            except RuntimeError as e:
                oom_strings = [
                    "out of memory", 
                    "cuda out of memory",
                    "alloc failed",
                    "memory allocation"
                ]
                
                if any(oms in str(e).lower() for oms in oom_strings):
                    logger.warning(f"OOM with batch size {args.batch_size}, processing one-by-one")
                    batch_oom = True
                else:
                    logger.error(f"Runtime error: {str(e)}")
                    raise
            
            if batch_oom:
                for i in range(current_index, end_index):
                    single_batch = df_chunk.iloc[i:i+1]
                    seq_index = single_batch.index[0]
                    sequence = single_batch.iloc[0]['sequence']
                    cleaned_seq = single_batch.iloc[0]['cleaned_sequence']
                    seq_length = len(cleaned_seq)
                    
                    try:
                        single_results = process_batch(args, device, model, single_batch)
                        
                        if single_results:
                            result = single_results[0]
                            file_path, feature_path = feature_writer.add_feature(
                                result['s_z'], result['pae'], result['contact']
                            )
                            all_results.append({
                                'sequence': result['sequence'],
                                'label': result['label'],
                                'affinity': result['affinity'],
                                'mhc': result['mhc'],
                                'peptide': result['peptide'],
                                'feature_file': os.path.basename(file_path),
                                'feature_path': feature_path
                            })
                        
                        pbar.update(1)
                        progress_queue.put(1)
                        
                    except RuntimeError as e2:
                        if any(oms in str(e2).lower() for oms in oom_strings):
                            error_msg = (f"OOM on single sequence at index {seq_index}, "
                                        f"length={seq_length}, sequence: {sequence}")
                            logger.error(error_msg)
                            
                            with open(skipped_file, "a") as f:
                                f.write(f"{seq_index},{sequence},{seq_length},OOM\n")
                            
                            skipped_count += 1
                        else:
                            logger.error(f"Runtime error processing single sequence: {str(e2)}")
                            raise
                    except Exception as e2:
                        logger.error(f"Unexpected error processing single sequence: {str(e2)}")
                        raise
                
                torch.cuda.empty_cache()
                gc.collect()
                
                current_index = end_index
    
    except Exception as e:
        logger.error(f"Error in shard {shard_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        feature_writer.close()
        pbar.close()
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} sequences due to OOM errors")
    
    out_path = f"{args.output_csv}.part{shard_id}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(all_results)} metadata to {out_path}")
        result_queue.put(out_path)
    else:
        logger.warning(f"No valid results for shard {shard_id}")
        result_queue.put(None)
    
    logger.info(f"Shard {shard_id} finished")
    progress_queue.put(-1)
    
    return skipped_count


def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='ESMFold Feature Extraction with HDF5 Storage')
    parser.add_argument('--input_csv', required=True, help='Input CSV file with sequences')
    parser.add_argument('--output_csv', required=True, help='Output metadata CSV file')
    parser.add_argument('--feature_dir', default='features', help='Directory for HDF5 feature files')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--recycle', type=int, default=0, help='Number of recycling steps')
    parser.add_argument('--linker_len', type=int, default=25, help='Linker length between chains')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size for model')
    parser.add_argument('--group_size', type=int, default=1000, 
                       help='Number of samples per HDF5 file')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), 
                       help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.feature_dir, exist_ok=True)
    
    main_log = os.path.join(args.feature_dir, "feature_extraction_main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(main_log)
        ]
    )
    logger = logging.getLogger()
    
    logger.info(f"Starting feature extraction with parameters:")
    logger.info(f"  Input CSV: {args.input_csv}")
    logger.info(f"  Output CSV: {args.output_csv}")
    logger.info(f"  Feature directory: {os.path.abspath(args.feature_dir)}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Group size: {args.group_size}")
    logger.info(f"  Workers: {args.workers}")
    
    try:
        df = pd.read_csv(args.input_csv)
        total_sequences = len(df)
        logger.info(f"Loaded {total_sequences} sequences from {args.input_csv}")
        
        df['cleaned_sequence'] = df['sequence'].apply(
            lambda x: clean_seq(x) if pd.notnull(x) else ""
        )
        df['length'] = df['cleaned_sequence'].apply(len)
        max_length = df['length'].max()
        avg_length = df['length'].mean()
        logger.info(f"Sequence length statistics: Max={max_length}, Avg={avg_length:.2f}")
    except Exception as e:
        logger.error(f"Failed to read input CSV: {str(e)}")
        return
    
    progress_queue = mp.Queue()
    progress_monitor = ProgressMonitor(total_sequences)
    
    shard_size = total_sequences // args.workers
    shards = []
    for i in range(args.workers):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < args.workers - 1 else total_sequences
        shard_df = df.iloc[start_idx:end_idx].copy()
        shard_df['original_index'] = shard_df.index
        shards.append(shard_df)
        logger.info(f"Shard {i}: {start_idx}-{end_idx} ({len(shards[-1])} samples)")
    
    processes = []
    result_queue = mp.Queue()
    skipped_counts = mp.Array('i', [0] * args.workers)
    
    for i in range(args.workers):
        p = mp.Process(
            target=process_shard,
            args=(i, shards[i], args, args.feature_dir, result_queue, progress_queue)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started shard {i} (PID: {p.pid})")
    
    def progress_listener():
        active_shards = args.workers
        while active_shards > 0:
            try:
                progress = progress_queue.get(timeout=30)
                if progress == -1:
                    active_shards -= 1
                else:
                    progress_monitor.update(progress)
            except:
                pass
    
    listener_thread = threading.Thread(target=progress_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    total_skipped = 0
    for i, p in enumerate(processes):
        p.join()
        exitcode = p.exitcode
        if exitcode == 0:
            logger.info(f"Shard process {p.pid} finished successfully")
        else:
            logger.error(f"Shard process {p.pid} failed with exitcode {exitcode}")
    
    progress_monitor.stop()
    listener_thread.join(timeout=1)
    
    output_files = []
    for _ in range(args.workers):
        result = result_queue.get()
        if result:
            output_files.append(result)
            logger.info(f"Received result file: {result}")
    
    all_dfs = []
    for file in output_files:
        if os.path.exists(file):
            try:
                df_part = pd.read_csv(file)
                all_dfs.append(df_part)
                logger.info(f"Loaded {len(df_part)} records from {file}")
            except Exception as e:
                logger.error(f"Error reading {file}: {str(e)}")
    
    skipped_files = glob.glob(os.path.join(args.feature_dir, "shard*_oom_sequences.csv"))
    oom_skipped = 0
    for file in skipped_files:
        try:
            oom_skipped += sum(1 for line in open(file)) - 1
        except:
            pass
    
    processed_count = sum(len(df) for df in all_dfs) if all_dfs else 0
    skipped_invalid = total_sequences - processed_count - oom_skipped
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df['feature_dir'] = os.path.abspath(args.feature_dir)
        final_df.to_csv(args.output_csv, index=False)
        logger.info(f"Merged {len(final_df)} records to {args.output_csv}")
    else:
        logger.error("No results to merge")
        return
    
    max_length = int(max_length)
    oom_skipped = int(oom_skipped)
    skipped_invalid = int(skipped_invalid)
    processed_count = int(processed_count)
    
    feature_files = glob.glob(os.path.join(args.feature_dir, "features_shard*_group*.h5"))
    summary = {
        "total_samples": int(total_sequences),
        "processed_samples": int(len(final_df)),
        "skipped_samples": {
            "oom": oom_skipped,
            "invalid_sequence": skipped_invalid,
            "total": oom_skipped + skipped_invalid
        },
        "feature_files": int(len(feature_files)),
        "feature_dir": os.path.abspath(args.feature_dir),
        "metadata_file": os.path.abspath(args.output_csv),
        "completed_at": datetime.now().isoformat(),
        "max_sequence_length": int(max_length)
    }
    
    summary_file = os.path.join(args.feature_dir, "processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing complete. Summary saved to {summary_file}")
    logger.info(f"Total sequences: {total_sequences}")
    logger.info(f"Processed sequences: {len(final_df)}")
    logger.info(f"Skipped due to OOM: {oom_skipped}")
    logger.info(f"Skipped due to invalid sequences: {skipped_invalid}")
    logger.info(f"Total skipped sequences: {oom_skipped + skipped_invalid}")
    logger.info(f"Max sequence length: {max_length}")
    
    if oom_skipped > 0:
        logger.warning(f"OOM skipped sequences recorded in: {', '.join(skipped_files)}")
    logger.info("="*50)
    logger.info("Feature extraction completed successfully!")

if __name__ == '__main__':
    main()

