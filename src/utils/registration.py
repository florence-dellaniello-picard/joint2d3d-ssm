"""
Author: Florence Dell'Aniello Picard
Registration utilities: register images to a reference template and save the resulting velocity fields.
"""

from pathlib import Path
import torch
import numpy as np


def register_velocities(models, dataset, reference, view, output_dir, save_folder, device):
    """Register each image in a dataset to the reference template and save the resulting velocity fields."""

    vel_dir = Path(output_dir) / 'velocities'
    
    split_vel_dir = vel_dir / save_folder / view
    split_vel_dir.mkdir(parents=True, exist_ok=True)
    model = models[view].to(device).eval()
    for i, moving_path in enumerate(dataset.paths):
        try:      
            moving = dataset._load(moving_path).unsqueeze(0).to(device)
            fixed = dataset._load(reference).unsqueeze(0).to(device)
            with torch.no_grad():
                flow = model(moving, fixed, return_field_type='svf')
                flow_np = flow.squeeze(0).cpu().detach().numpy()
            
            moving_id = Path(moving_path).stem.replace('.nii', '')
            vel_path = split_vel_dir / f'{moving_id}.npy'          
            np.save(vel_path, flow_np)

        except Exception as e:
            print(f"  Failed to register {moving_path}: {e}")
            continue        
