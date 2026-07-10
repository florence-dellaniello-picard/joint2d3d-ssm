"""
Author: Florence Dell'Aniello Picard
Model utilities: build the per-view VoxelMorph registration models and load their pre-trained checkpoints.
"""

from pathlib import Path
import torch
import torch.nn as nn
import voxelmorph.nn.models as models

def define_models(config, device, views=('ap', 'ml', '3d')):
    """
    Define the models for loading.
    """
    vxm_models = {}
    for view in views:
        p = config["vxm_models"].get("model_params", {})
        m = config["vxm_models"]["features"]

        # Only ndim and nb_features differ between 2D and 3D
        ndim       = 3 if view == "3d" else 2
        nb_features = m["3d"] if view == "3d" else m["2d"]

        model = models.VxmPairwise(
            ndim                     = p.get("ndim",                     ndim),
            source_channels          = p.get("source_channels",          1),
            target_channels          = p.get("target_channels",          1),
            nb_features              = nb_features,
            activations              = p.get("activations",              nn.ReLU),
            final_activation         = p.get("final_activation",         None),
            flow_initializer         = p.get("flow_initializer",         1e-5),
            integration_steps        = p.get("integration_steps",        5),
            resize_integrated_fields = p.get("resize_integrated_fields", False),
            unet_kwargs              = p.get("unet_kwargs",              False),
            device                   = device,
        ).to(device)

        vxm_models[view] = model

    return vxm_models

def load_checkpoints(config, vxm_models, train_size, device, views=('ap', 'ml', '3d')):
    """Load pre-trained model checkpoints for all views."""
    vxm_checkpoints = {}

    for view in views:
        vxm_path = Path(config['checkpoints_dir']) / config['vxm_models']['checkpoint_name'].format(view=view, train_size=train_size)
        if vxm_path.exists():
            ckpt = torch.load(vxm_path, map_location=device, weights_only=False)
            vxm_models[view].load_state_dict(ckpt['model_state'])
            vxm_models[view].eval()
            vxm_checkpoints[view] = ckpt
        else:
            print(f"[vxm/{view}] Checkpoint not found at {vxm_path}")
    
    return vxm_checkpoints