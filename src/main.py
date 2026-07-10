"""
Author: Florence Dell'Aniello Picard
Build the joint 2D-3D statistical shape model (SSM) from 2D and 3D SVF and/or perform patient-specific shape reconstruction from 2D images alone.
"""

import argparse
from pathlib import Path
import time
from utils.config import load_config
from train import run_training
from inference import run_inference

def parse_args():
    parser = argparse.ArgumentParser(
        description='Joint 2D-3D SSM pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument('--config', default='configs/joint2d3d-ssm.yaml',
                        help='Path to YAML config file (default: configs/joint2d3d-ssm.yaml)')
    parser.add_argument('--id', default=-1, type=int,
                        help='Slurm array id corresponding to patient id')
    parser.add_argument('--incremental', default=-1, type=int,
                        help='Number of patients per Slurm chunk')
    parser.add_argument('--run', type=str, choices=['train', 'inference', 'both'], default='both',
                        help='Whether to run training, inference or both')
    
    # Path overrides
    parser.add_argument('--root_input', default=None,
                        help='Override config root path')
    parser.add_argument('--root_output', default=None,
                        help='Override config root_vel path')

    # Inference parameter overrides
    parser.add_argument('--reg-strength', type=float, default=None, dest='reg_strength',
                        help='Override encode_2d reg_strength')
    parser.add_argument('--variance-target', type=float, default=None, dest='variance_target',
                        help='Override PCA variance target for mode selection')

    return parser.parse_args()


if __name__ == '__main__':
    print("\n" + "="*150)
    print('JOINT 2D-3D SSM')

    args   = parse_args()
    config = load_config(args.config, input_override=args.root_input, output_override=args.root_output, patient_id=args.id)

    # Apply CLI overrides
    config['incremental'] = args.incremental
    config['run'] = args.run
    if args.reg_strength is not None:
        config['joint_2d3d_ssm']['inference']['reg_strength'] = args.reg_strength
    if args.variance_target is not None:
        config['joint_2d3d_ssm']['inference']['variance_target'] = args.variance_target

    print("-"*150)

    run = config['run']

    output_dir = Path(config['output_root']) / 'ssm'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[joint_ssm] Output directory: {output_dir}')

    if run in ('train', 'both'):
        print('[joint_ssm] Joint 2D-3D Statistical Shape Model (Training)')
        t0 = time.perf_counter()
        run_training(config, output_dir)
        print(f'            Elapsed time to build joint 2D-3D SSM: {time.perf_counter() - t0:.2f}s')

    if run in ('inference', 'both'):
        print('[joint_ssm] Patient-Specific Shape Reconstruction (Inference)')
        run_inference(config, output_dir)

    print('[pipeline] Done.')