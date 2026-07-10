"""
Author: Florence Dell'Aniello Picard
Training phase: register SVFs for each modality and fit the joint 2D-3D SSM.
"""

from pathlib import Path
import numpy as np
import torch

from data_loader.data_loader import load_datasets
from models.models import define_models, load_checkpoints
from utils.registration import register_velocities
from joint2d3d_ssm import joint_SSM


def run_training(config, output_dir, views=('ap', 'ml', '3d')):
    """Register SVFs for all modalities and fit the joint 2D-3D PCA model."""
    device = torch.device(
        config["joint_2d3d_ssm"]["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    split = config['joint_2d3d_ssm']['training']['split']

    print("            Loading datasets...")
    datasets = load_datasets(config, device, views=views, subset_size=None)

    train_size = len(datasets[views[0]]['registration'])

    print("            Loading models...")
    models = define_models(config, device, views=views)
    load_checkpoints(config, models, train_size, device, views=views)

    # Computing velocities for each modality (skip views already registered)
    print("            Computing velocities...")
    vxm_vel_dir = output_dir / 'velocities' / split
    for view in views:
        (vxm_vel_dir / view).mkdir(exist_ok=True, parents=True) # Create directory if needed
        
        if any((vxm_vel_dir / view).glob('*.npy')):
            print(f"            [{view}] velocity fields already exist, skipping registration")
            continue

        template = Path(config['templates_dir']) / f'template_{view}.nii.gz'
        register_velocities(
            models, datasets[view][split],
            template, view, output_dir, split, device
        )

    # Load velocity paths
    vel_files_ap = sorted((vxm_vel_dir / views[0]).glob('*.npy'))
    vel_files_ml = sorted((vxm_vel_dir / views[1]).glob('*.npy'))
    vel_files_3d = sorted((vxm_vel_dir / views[2]).glob('*.npy'))

    if not vel_files_ap or not vel_files_ml or not vel_files_3d:
        raise FileNotFoundError(f"Velocity files not found in {vxm_vel_dir}")

    # Fit PCA
    pca_dir  = output_dir / 'joint_pca'
    pca_dir.mkdir(exist_ok=True, parents=True)
    pca_path = pca_dir / 'joint_2d3d_pca.npz'

    if pca_path.exists():
        print(f'            PCA already exists at {pca_path}, skipping...')
        return

    print(f"            Building joint 2D-3D SSM from {len(vel_files_ap)} velocity fields in {vxm_vel_dir}...")
    pca = joint_SSM(
        n_components=config['joint_2d3d_ssm']['training']['n_modes'],
        random_seed=config['joint_2d3d_ssm']['training']['random_seed'],
    )
    pca.fit(
        paths_ap = vel_files_ap,
        paths_ml = vel_files_ml,
        paths_3d = vel_files_3d,
        shape_ap = np.load(vel_files_ap[0]).shape,
        shape_ml = np.load(vel_files_ml[0]).shape,
        shape_3d = np.load(vel_files_3d[0]).shape,
        tmp_dir  = output_dir / 'tmp',
    )

    print(f"            Total components: {pca.n_components}")
    print(f"            Total variance explained: {pca.cumulative_variance_ratio_[-1]:.2%}")
    for thresh in [0.90, 0.95, 0.99]:
        idx = np.argmax(pca.cumulative_variance_ratio_ >= thresh)
        if pca.cumulative_variance_ratio_[idx] >= thresh:
            print(f"            Components needed for {thresh:.0%}: {idx + 1}")

    pca.save(pca_path)
