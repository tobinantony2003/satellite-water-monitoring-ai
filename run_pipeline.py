"""
Integrated Deep-Learning Water Extraction and Water Quality Inversion Pipeline
Master Pipeline Runner

Usage:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage eval-only
    python run_pipeline.py --stage 1
    python run_pipeline.py --stage 2
    python run_pipeline.py --stage 3
    python run_pipeline.py --stage 4
"""

import os
import sys

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings('ignore', message='.*physical cores.*')
warnings.filterwarnings('ignore', category=UserWarning, module='.*loky.*')

import argparse
import time
import pandas as pd


def check_environment_and_directories():
    """Verifies directory structures and input data paths."""
    print("=" * 70)
    print("SATELLITE WATER MONITORING AI: PIPELINE INITIALIZATION")
    print("Targeting: Remote Sensing of Environment / Int. J. Remote Sensing")
    print("=" * 70)
    
    dirs = [
        'results/checkpoints',
        'results/figures',
        'results/tables',
        'data/processed/matched_wq_cleaned',
        'reports'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    dset_s2_path = os.path.join('data', 'raw', 'dset-s2', 'dset-s2')
    matched_wq_path = os.path.join('data', 'raw', 'matched_wq_s2', 'Matched_WQ_S2')
    
    if not os.path.exists(dset_s2_path):
        print(f"Warning: dset-s2 directory not found at {dset_s2_path}", flush=True)
    else:
        print(f"[OK] Found dset-s2 benchmark dataset at {dset_s2_path}", flush=True)
        
    if not os.path.exists(matched_wq_path):
        print(f"Warning: Matched_WQ_S2 directory not found at {matched_wq_path}", flush=True)
    else:
        print(f"[OK] Found USGS Matched_WQ_S2 dataset at {matched_wq_path}", flush=True)
    print("=" * 70, flush=True)

def run_stage_1(retrain: bool = False, epochs: int = 5):
    """Stage 1: Deep Learning Water Extraction Front-End."""
    print("\n" + "#" * 70)
    print("STAGE 1: Deep Learning Water Extraction Model (Water U-Net)")
    print("#" * 70, flush=True)
    
    ckpt_path = os.path.join('results', 'checkpoints', 'water_unet_best.pt')
    if retrain or not os.path.exists(ckpt_path):
        from src.segmentation.train import train_segmentation
        print(f"Executing full training of Water U-Net ({epochs} epochs)...", flush=True)
        train_segmentation(epochs=epochs, batch_size=16)
    else:
        from src.segmentation.run_eval import evaluate_saved_model
        print(f"Evaluating validated checkpoint: {ckpt_path}...", flush=True)
        evaluate_saved_model(checkpoint_path=ckpt_path)
        
    print("[OK] Stage 1 Completed Successfully!", flush=True)

def run_stage_2(sample_size: int = 150000):
    """Stage 2: Water Masking, Parallel Datasets, & Disagreement Analysis."""
    print("\n" + "#" * 70)
    print("STAGE 2: Water Mask Application & Parallel Dataset Construction")
    print("#" * 70, flush=True)
    
    from src.inversion.build_matched_sets import generate_parallel_datasets
    from src.inversion.eda import run_exploratory_analysis
    
    print("Generating parallel datasets: (a) Assumed, (b) DL Mask, (c) NDWI Mask...", flush=True)
    df_a, df_b, df_c, df_disagree = generate_parallel_datasets(
        sample_size=sample_size,
        save_dir='data/processed/matched_wq_cleaned',
        results_dir='results'
    )
    
    print("\nGenerating publication EDA figures and spectral curves...", flush=True)
    run_exploratory_analysis(df_b, results_dir='results')
    print("[OK] Stage 2 Completed Successfully!", flush=True)

def run_stage_3():
    """Stage 3: Water Quality Inversion Modeling with Spatial Block CV."""
    print("\n" + "#" * 70)
    print("STAGE 3: Inversion Model Suite with 5-Fold Spatial Block CV")
    print("#" * 70, flush=True)
    
    from src.inversion.train_inversion import benchmark_inversion_models
    
    dl_path = os.path.join('data', 'processed', 'matched_wq_cleaned', 'matched_set_b_dl_masked.csv')
    if not os.path.exists(dl_path):
        raise FileNotFoundError(f"Missing {dl_path}. Please execute Stage 2 first.")
        
    df_dl = pd.read_csv(dl_path, low_memory=False)
    
    consolidated = []
    for target in ['turbidity', 'chlorophyll', 'cdom']:
        res_df, _ = benchmark_inversion_models(
            df_dl,
            target_type=target,
            dataset_name='Condition_(b)_DL_Masked',
            results_dir='results'
        )
        consolidated.append(res_df)
        
    cons_df = pd.concat(consolidated, ignore_index=True)
    cons_path = os.path.join('results', 'tables', 'stage3_consolidated_inversion_benchmarks.csv')
    cons_df.to_csv(cons_path, index=False)
    print(f"\n[OK] Consolidated Stage 3 Inversion Benchmarks saved to {cons_path}", flush=True)
    print("[OK] Stage 3 Completed Successfully!", flush=True)

def run_stage_4():
    """Stage 4: Core Scientific Experiments (Ablation, Portability, Temporal)."""
    print("\n" + "#" * 70)
    print("STAGE 4: Core Experiments & Statistical Validation")
    print("#" * 70, flush=True)
    
    from src.evaluation.ablation import run_experiment_a_ablation
    from src.evaluation.transferability import run_experiment_b_transferability
    from src.evaluation.temporal import run_experiment_c_temporal
    
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    # Experiment A: Segmentation-Quality Ablation & Statistical Testing
    print("\n--- Running Experiment A: Masking Ablation & Statistical Tests ---", flush=True)
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_a_ablation(df_a, df_b, df_c, target_type=target, model_name='Gradient_Boosting')
        
    # Experiment B: Cross-Basin Portability & Transferability
    print("\n--- Running Experiment B: Cross-Basin Portability ---", flush=True)
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_b_transferability(df_b, target_type=target, model_name='Gradient_Boosting')
        
    # Experiment C: Multi-Year Temporal Robustness
    print("\n--- Running Experiment C: Temporal Robustness ---", flush=True)
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_c_temporal(df_b, target_type=target, model_name='Gradient_Boosting')
        
    print("[OK] Stage 4 Completed Successfully!", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Satellite Water Monitoring AI Pipeline Runner")
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['all', '1', '2', '3', '4', 'eval-only', 'quick-eval'],
        help="Pipeline stage to execute: 'all', '1', '2', '3', '4', or 'eval-only'"
    )
    parser.add_argument('--retrain-stage1', action='store_true', help="Force retraining of Stage 1 U-Net")
    parser.add_argument('--epochs-stage1', type=int, default=5, help="Number of epochs if retraining Stage 1")
    parser.add_argument('--sample-size-stage2', type=int, default=150000, help="Sample size for Stage 2 parsing")
    
    args = parser.parse_args()
    
    start_time = time.time()
    check_environment_and_directories()
    
    stage = args.stage
    if stage in ['all', '1']:
        run_stage_1(retrain=args.retrain_stage1, epochs=args.epochs_stage1)
        
    if stage in ['all', '2']:
        run_stage_2(sample_size=args.sample_size_stage2)
        
    if stage in ['all', '3']:
        run_stage_3()
        
    if stage in ['all', '4']:
        run_stage_4()
        
    if stage in ['eval-only', 'quick-eval']:
        run_stage_1(retrain=False)
        run_stage_3()
        run_stage_4()
        
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE EXECUTION FINISHED SUCCESSFULLY IN {elapsed:.1f}s (EXIT CODE 0)")
    print("All validation tables in results/tables/ and figures in results/figures/")
    print("=" * 70 + "\n", flush=True)

if __name__ == '__main__':
    main()
