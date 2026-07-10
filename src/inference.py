"""
Author: Florence Dell'Aniello Picard
Inference phase: encode 2D SVFs into the joint latent space and reconstruct 3D shape.
"""

import pickle
import time
from pathlib import Path
import voxelmorph as vxm
import numpy as np
import pandas as pd
import torch
from utils.registration import register_velocities
from utils.metrics import dice_score, hausdorff_and_msd
from data_loader.data_loader import load_datasets, NMDIDLoader
from models.models import define_models, load_checkpoints
from joint2d3d_ssm import joint_SSM

def get_patient_ids(report_path, split):
    """Get the sorted list of '{patient_id}_L'/'{patient_id}_R' IDs usable for a given split."""
    df = pd.read_csv(report_path)
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    col_L = 'excluded_L'
    col_R = 'excluded_R'

    split_mask = df['split'].str.lower() == split.lower()
    df = df[split_mask]

    ids = []
    for _, row in df.iterrows():
        if col_L in df.columns and str(row[col_L]).lower() == 'no':
            ids.append(f"{row['patient_id']}_L")
        if col_R in df.columns and str(row[col_R]).lower() == 'no':
            ids.append(f"{row['patient_id']}_R")

    return sorted(ids)

def run_inference(config, output_dir, views=('ap', 'ml', '3d')):
    """Encode 2D AP+ML SVFs and reconstruct 3D shape; evaluate against GT segmentations."""
    device = torch.device(
        config["joint_2d3d_ssm"]["inference"]["device"] if torch.cuda.is_available() else "cpu"
    )
    split = config['joint_2d3d_ssm']['inference']['split']

    print("            Loading datasets...")
    datasets   = load_datasets(config, device, views=views)
    loader     = NMDIDLoader(view='3d', target_shape=config['target_shape']['3d'])
    train_size = len(datasets[views[0]]['registration'])

    print("            Loading models...")
    models = define_models(config, device, views=views)
    load_checkpoints(config, models, train_size, device, views=views)

    pca_path = output_dir / 'joint_pca' / 'joint_2d3d_pca.npz'
    pca      = joint_SSM.load(pca_path)

    variance_target = config['joint_2d3d_ssm']['inference']['variance_target']
    n_modes         = int(np.argmax(pca.cumulative_variance_ratio_ >= variance_target)) + 1
    reg_strength    = config['joint_2d3d_ssm']['inference']['reg_strength']
    print(f"            [pca] {n_modes} components needed for {variance_target:.0%} variance; reg_strength={reg_strength}")

    # Load the 3D template (for metric computation)
    template_path = Path(config['templates_dir']) / 'template_3d.nii.gz'
    template_np   = loader(template_path).squeeze().numpy()

    # Patient list
    img_ids = sorted(get_patient_ids(
            config['report_path'], config['joint_2d3d_ssm']['inference']['split']
    ))

    # Process patient batches
    patient_id   = config.get('patient_id', -1)
    incremental  = config.get('incremental', 0)
    if patient_id == -1:
        ids_to_run = img_ids
    elif incremental > 0:
        start      = patient_id * incremental
        ids_to_run = img_ids[start: start + incremental]
        print(f'            [eval] Chunk {patient_id}: subjects {start}–{start + len(ids_to_run) - 1}')
    else:
        ids_to_run = [img_ids[patient_id]]

    seg_dir           = config['images_dir']
    pred_vel_save_dir = output_dir / 'predicted_velocities' / '3d'
    pred_vel_save_dir.mkdir(parents=True, exist_ok=True)
    eval_dir          = output_dir / 'pkl_results'
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = []

    def process_patient(subject_id):
        seg_path = Path(seg_dir) / f'{subject_id}.nii.gz'
        if not seg_path.exists():
            print(f'            [eval] Warning: no segmentation for {subject_id}, skipping')
            return

        gt_seg_np = loader(seg_path).squeeze().numpy()

        # Register AP and ML for this subject
        vxm_vel_dir = output_dir / 'velocities' / split

        t_reg_start = time.perf_counter()
        for view in views[0:2]:
            (vxm_vel_dir / view).mkdir(exist_ok=True, parents=True)

            template     = Path(config['templates_dir']) / f'template_{view}.nii.gz'

            dataset   = datasets[view][split]
            all_paths = sorted(dataset.paths)
            all_stems = [p.stem.replace('.nii', '') for p in all_paths]
            if subject_id not in all_stems:
                print(f'            [eval] Warning: {view} dataset has no entry for {subject_id}, skipping')
                return

            # Restrict the dataset to just this subject before registering
            original_paths = dataset.paths
            dataset.paths  = [all_paths[all_stems.index(subject_id)]]

            # Register to view-specific template
            register_velocities(
                models, dataset,
                template, view, output_dir, split, device
            )
            dataset.paths = original_paths

            t_elapsed_reg = time.perf_counter() - t_reg_start

        t_infer_start = time.perf_counter()
        # Reconstruct 3D from 2D SVFs
        ap_path = vxm_vel_dir / views[0]  / f'{subject_id}.npy'
        ml_path = vxm_vel_dir / views[1] / f'{subject_id}.npy'

        ap_vel  = np.load(ap_path)
        ml_vel = np.load(ml_path)
        z             = pca.encode_2d(ap_vel, ml_vel, n_components=n_modes, reg_strength=reg_strength)
        reconstructed = pca.reconstruct(z, n_components=n_modes)
        predicted_vel_field_3d = reconstructed['3d']
        np.save(pred_vel_save_dir / f'{subject_id}.npy', predicted_vel_field_3d)
        t_elapsed_infer = time.perf_counter() - t_infer_start
        t_elapsed = t_elapsed_reg + t_elapsed_infer

        t_warp_start = time.perf_counter()
        
        # Warp and compute metrices
        with torch.no_grad():
                predicted_vel_field_tensor = torch.from_numpy(predicted_vel_field_3d).float().unsqueeze(0).to(device)
                disp_from_predicted        = vxm.nn.functional.integrate_disp(-predicted_vel_field_tensor, steps=7)
                disp_from_predicted_np     = disp_from_predicted.squeeze(0).cpu().detach().numpy()

                # Warp the atlas with the predicted 3D SVF
                seg_tensor   = torch.from_numpy(template_np).float().unsqueeze(0).unsqueeze(0).to(device)
                disp_tensor  = torch.from_numpy(disp_from_predicted_np).float().unsqueeze(0).to(device)
                warped = vxm.nn.functional.spatial_transform(seg_tensor, disp_tensor, method='bilinear')
                predicted_np = warped.squeeze().cpu().numpy()
                del predicted_vel_field_tensor, disp_from_predicted

        t_elapsed_warp = time.perf_counter() - t_warp_start
        t_elapsed += t_elapsed_warp

        # Comparison 1: GT vs. 3D template (mean shape (lower bound))
        dice_template, hd_template, msd_template = (
            dice_score(template_np, gt_seg_np),
            *hausdorff_and_msd(template_np, gt_seg_np),
        )
        # Comparison 2: GT vs. predicted-warped (joint 2D-3D SSM)
        dice_predicted, hd_predicted, msd_predicted = (
            dice_score(predicted_np, gt_seg_np),
            *hausdorff_and_msd(predicted_np, gt_seg_np),
        )

        print(f'            [eval] {subject_id} Template  — Dice: {dice_template:.4f} | HD: {hd_template:.4f} | MSD: {msd_template:.4f}')
        print(f'            [eval] {subject_id} Predicted — Dice: {dice_predicted:.4f} | HD: {hd_predicted:.4f} | MSD: {msd_predicted:.4f} | time: {t_elapsed:.1f}s')

        result = {
            'subject_id': subject_id,
            'z_opt': z,
            'dice_template': dice_template, 'hd_template': hd_template, 'msd_template': msd_template,
            'dice_predicted': dice_predicted, 'hd_predicted': hd_predicted, 'msd_predicted': msd_predicted,
            'time_s': t_elapsed,
        }
        results.append(result)

        with open(eval_dir / f'{subject_id}.pkl', 'wb') as f:
                pickle.dump(result, f)

        if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for idx, subject_id in enumerate(ids_to_run):
        print(f'\n[eval] Subject {idx + 1}/{len(img_ids)}: {subject_id}')
        try:
                process_patient(subject_id)
        except Exception as e:
                print(f'[eval] Error processing {subject_id}: {e}')
                import traceback; traceback.print_exc()

        if not results:
                print('[eval] No subjects evaluated.')
