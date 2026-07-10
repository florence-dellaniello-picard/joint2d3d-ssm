"""
Author: Florence Dell'Aniello Picard
Data loading utilities: load DRRs/volumes from the NMDID dataset and build the per-view PyTorch datasets used for registration.
"""

from torch.utils.data import Dataset
from pathlib import Path

import os
import csv
import tempfile
import numpy as np
import torch
import nibabel as nib
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from PIL import Image
from pathlib import Path


class NMDIDLoader:
    """
    Handles all file loading for the NMDID dataset.
    Callable on a path to load a single sample based on view.
    """
    def __init__(self, view, views=('ap', 'ml'), target_shape=None, drr_params=None, device='cpu'):
        self.view         = view
        self.target_shape = target_shape
        self.device       = device
        self.drr_params   = drr_params

        # Initialize attributes that might be set conditionally
        self.bone_atten = None
        self.drr_parameters = None
        self.pose = None
        
        if view in views and drr_params:
            self.bone_atten       = drr_params['bone_attenuation']
            p                     = drr_params['parameters']
            subsample             = p['subsample']
            self.drr_parameters   = {
                'delx'  : float(p['delx']) * subsample,
                'height': int(p['height']) // subsample,
                'width' : int(p['width'])  // subsample,
                'x0'    : p['x0'],
                'y0'    : p['y0'],
                'sdd'   : p['sdd'],
            }
            self.pose_orientation = view
            self.pose             = self._build_pose()

    def __call__(self, path):
        """Load a single sample from disk."""
        if self.view == '3d':
            return self._load_volume(path)
        return self._load_drr(path)


    def _resize_volume(self, vol):
        """Resize a 3D volume to target_shape via bounding-box crop, center-crop, and zero-padding."""
        td, th, tw = (int(x) for x in self.target_shape)

        nz = np.argwhere(vol > 0)
        if len(nz) == 0:
            return np.zeros((td, th, tw), dtype=np.float32)

        d_min, h_min, w_min = nz.min(axis=0)
        d_max, h_max, w_max = nz.max(axis=0)
        vol = vol[d_min:d_max+1, h_min:h_max+1, w_min:w_max+1]

        d, h, w = vol.shape
        vol = vol[
            max((d - td) // 2, 0) : max((d - td) // 2, 0) + td,
            max((h - th) // 2, 0) : max((h - th) // 2, 0) + th,
            max((w - tw) // 2, 0) : max((w - tw) // 2, 0) + tw,
        ]

        def pad_axis(current, target):
            diff   = target - current
            before = diff // 2
            return before, diff - before

        return np.pad(
            vol,
            [pad_axis(s, t) for s, t in zip(vol.shape, (td, th, tw))],
            mode='constant',
        )
    
    def _load_volume(self, path):
        """Load a .nii.gz CT volume and optionally resize it."""
        vol = nib.load(path).get_fdata(dtype=np.float32)
        if self.target_shape is not None:
            vol = self._resize_volume(vol)
        return torch.from_numpy(vol[np.newaxis].astype(np.float32))

    def _build_pose(self):
        """Build the camera pose for DRR rendering from the config."""
        sdd      = self.drr_parameters['sdd']
        pose_cfg = self.drr_params.get('pose', {})
        pose_cfg = pose_cfg[self.view]
        
        t_gt, r_gt = self._parse_pose_cfg(sdd, pose_cfg)

        t_t = torch.tensor(t_gt, dtype=torch.float32, device=self.device).unsqueeze(0)
        r_t = torch.tensor(r_gt, dtype=torch.float32, device=self.device).unsqueeze(0)

        return convert(r_t, t_t, parameterization='euler_angles', convention='ZYX')

    def _parse_pose_cfg(self, sdd, pose_cfg):
        """Parse translation and rotation from a pose config block."""
        t_gt = np.array([
            eval(str(v), {'sdd': sdd}) if isinstance(v, str) else v
            for v in pose_cfg['translation']
        ], dtype=float)
        r_gt = np.deg2rad(np.array(pose_cfg['rotation_deg'], dtype=float))
        return t_gt, r_gt
    

    def _crop_to_bbox(self, path):
        """Crop a volume to the bounding box of its own foreground voxels and save to a temp file."""
        vol = nib.load(path).get_fdata(dtype=np.float32)

        nz = np.argwhere(vol > 0)

        if len(nz) == 0:
            return str(path)

        d_min, h_min, w_min = nz.min(axis=0)
        d_max, h_max, w_max = nz.max(axis=0)
        cropped = vol[d_min:d_max+1, h_min:h_max+1, w_min:w_max+1]

        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        nib.save(nib.Nifti1Image(cropped, np.eye(4)), tmp.name)
        return tmp.name
    
    def _load_drr(self, path):
        """Render a DRR from a CT volume, normalized to [0, 1]."""
        if Path(path).suffix.lower() == ".png":
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            arr = arr / 255.0
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        
        vol = nib.load(path).get_fdata(dtype=np.float32)
        if vol.ndim == 3 and vol.shape[0] == 1:
            arr = vol[0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        cropped_path = self._crop_to_bbox(path)
        try:
            subject  = read(cropped_path, bone_attenuation_multiplier=self.bone_atten)
            renderer = DRR(subject=subject, **self.drr_parameters).to(self.device)

            with torch.no_grad():
                drr = renderer(self.pose).squeeze()  # (H, W)

            drr_min, drr_max = drr.min(), drr.max()
            drr = (drr - drr_min) / (drr_max - drr_min + 1e-8)
        finally:
            if cropped_path != str(path):
                os.unlink(cropped_path)

        return drr.cpu().unsqueeze(0)  # (1, H, W)

    @staticmethod
    def load_report(report_path):
        """Load dataset_report.csv into a list of row dicts."""
        with open(report_path, newline='') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def filter_usable(rows, side, split=None):
        """Filter report rows to usable samples for a given side."""
        key  = f'excluded_{side}'
        rows = [r for r in rows if r.get(key) == 'No']
        if split is not None:
            rows = [r for r in rows if r['split'] == split]
        return rows

    @staticmethod
    def resolve_path(root, pid, side):
        """
        Resolve the file path for a given patient and side.
        """
        return root / f'{pid}_{side}.nii.gz'


class NMDID_voxelmorph(Dataset):
    """
    PyTorch Dataset for pairwise registration on the NMDID dataset.
    """
    def __init__(self, report_path, root, view, split, target_shape, views=('ap', 'ml', '3d'), drr_params=None, device='cpu'):
        assert view in views, f"view must be 'ap', 'ml', or '3d', got '{view}'"
        self.view         = view
        self.root         = Path(root)
        self.target_shape = target_shape
        self._load        = NMDIDLoader(view, target_shape=target_shape, drr_params=drr_params, device=device)

        rows = NMDIDLoader.load_report(Path(report_path))
        self.paths = []
        for side in ('L', 'R'):
            side_rows = NMDIDLoader.filter_usable(rows, side, split)
            self.paths += [NMDIDLoader.resolve_path(self.root, r['patient_id'], side) for r in side_rows]

    def __len__(self):
        """Returns the number of usable samples."""
        return len(self.paths)


def load_datasets(config, device, views=('ap', 'ml', '3d'), subset_size=None):
    """Load train/test/eval datasets for each view (ap, ml, 3d)."""

    def maybe_slice(dataset):
        if subset_size is not None:
            dataset.paths = dataset.paths[:subset_size]
        return dataset

    def build_for_view(view):
        drr_params   = config["drr"]
        report_path  = config["report_path"]
        root         = config["images_dir"]
        shape_key    = "3d" if view == "3d" else "2d"
        target_shape = config["target_shape"][shape_key]
        args = (report_path, root, view)

        return {
            split: maybe_slice(
                NMDID_voxelmorph(*args, split, target_shape, drr_params=drr_params, device=device)
            )
            for split in ["registration", "shape_model", "val", "test"]
        }
    return {view: build_for_view(view) for view in views}