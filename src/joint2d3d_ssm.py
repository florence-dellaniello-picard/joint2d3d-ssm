"""
Author: Florence Dell'Aniello Picard
Joint 2D-3D SSM fit across AP, Lat, and 3D velocity fields and 2D-to-3D reconstruction.
"""

import os
import gc
import tempfile
import time
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import cho_factor, cho_solve


class joint_SSM:
    """
    Joint PCA over three views per subject:
        - 2D AP view   (paths_ap)
        - 2D ML view  (paths_ml)
        - 3D velocity  (paths_3d)
    """
   
    def __init__(self, n_components, random_seed=None):
        self.views                      = ('ap', 'ml', '3d')

        self.n_components               = n_components
        self.random_seed                = random_seed
        
        self.components_                = None   # (n_components, total_dim)
        self.explained_variance_ratio_  = None
        self.explained_variance_        = None
        self.cumulative_variance_ratio_ = None
        self.singular_values_           = None

        self.mean_           : Dict[str, np.ndarray] = {}
        self.std_            : Dict[str, np.ndarray] = {}
        self.original_shape_ : Dict[str, tuple]      = {}
        self.dim_            : Dict[str, int]        = {}
        self.eps_            : Dict[str, float]      = {}
        self.view_scale : Dict[str, float]           = {}
        
    def _boundaries(self):
        """Compute each view's start offset within the concatenated joint feature vector."""
        starts, cursor = {}, 0
        for key in self.views:
            starts[key] = cursor     
            cursor += self.dim_[key]  
        return starts, cursor      

    def _compute_mean_std(self, paths_ap, paths_ml, paths_3d):
        """Compute the per-voxel mean and standard deviation of each view."""
        stats = {}
        for key, paths in zip(self.views, [paths_ap, paths_ml, paths_3d]):
            n = len(paths)
            flat_dim = np.load(paths[0]).size
            mean     = np.zeros(flat_dim)

            # Accumulate the sum (avoid loading all at once)
            for path in paths:
                mean += np.load(path).ravel()
            mean /= n

            # Accumulate squared deviations from the mean
            M2 = np.zeros(flat_dim)
            for path in paths:
                diff  = np.load(path).ravel() - mean
                M2   += diff ** 2

            stats[key] = {
                "mean": mean,
                "std":  np.sqrt(M2 /  (n-1)),
            }
        return stats
    
    def _compute_eps(self):
        """Derive a per-view noise floor from std_."""
        eps = {}
        for key in self.views:
            std = self.std_.get(key)
            eps[key] = max(1e-8, float(np.percentile(std, 99)) * 1e-3)
        return eps
    
    def _normalize(self, x, key):
        """Z-score normalize a sample (zero near-constant voxels and clip outliers)."""
        x_flat   = x.ravel().astype(np.float32)
        mean_val = self.mean_[key]  
        std_val  = self.std_[key]   
        eps      = self.eps_[key]  

        # Z-score normalize (zero out near-constant voxels)
        safe_std = np.where(std_val < eps, np.inf, std_val)
        z        = (x_flat - mean_val) / safe_std

        active = std_val >= eps
        if active.any():
            # Adaptive clip: bound scales with each voxel's std relative to the median (typical ~3, range 1.5-10)
            median_std  = float(np.median(std_val[active]))
            local_scale = np.where(active, std_val / (median_std + 1e-9), 1.0)
            clip_val    = np.clip(3.0 * local_scale, 1.5, 10.0)
            z           = np.clip(z, -clip_val, clip_val)

        return z
    
    def _denormalize(self, x_norm, key):
        """Invert _normalize."""
        mean_val = self.mean_[key]
        std_val  = self.std_[key]
        eps      = self.eps_[key]
        
        # Denormalize
        std_safe = np.where(std_val < eps, 1.0, std_val)
        return x_norm * std_safe + mean_val

    def _compute_X_matrix(self, paths_ap, paths_ml, paths_3d, tmp_dir=None):
        """Normalize and concatenate every sample's AP/ML/3D fields."""
        n_samples = len(paths_ap)
        _, total  = self._boundaries()

        tmp_dir = tmp_dir or tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)

        for stale in Path(tmp_dir).glob('tmp*.npy'):
            stale.unlink()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npy', dir=tmp_dir)
        mmap_path = tmp.name
        tmp.close()

        try:
            # Allocate the (n_samples x total_dim) array on disk instead of in memory
            mmap_data = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_samples, total))
            for i, (pa, pm, p3) in enumerate(zip(paths_ap, paths_ml, paths_3d)):
                mmap_data[i] = np.concatenate([
                    self.view_scale['ap'] * self._normalize(np.load(pa), 'ap'),
                    self.view_scale['ml'] * self._normalize(np.load(pm), 'ml'),
                    self.view_scale['3d'] * self._normalize(np.load(p3), '3d'),
                ])

            mmap_data.flush()

        except Exception:
            os.unlink(mmap_path)
            raise

        return mmap_path
      
    def fit(self, paths_ap, paths_ml, paths_3d, shape_ap, shape_ml, shape_3d, tmp_dir=None):
        """Fit joint 2D-3D SSM via PCA on X (n_samples × total_dim)."""
        assert len(paths_ap) == len(paths_ml) == len(paths_3d)

        for key, shape in zip(self.views, [shape_ap, shape_ml, shape_3d]):
            self.original_shape_[key] = shape
            self.dim_[key]            = int(np.prod(shape))

        # Make sure n_components is at most (n_samples - 1)
        n_samples    = len(paths_ap)
        n_components = min(self.n_components, n_samples - 1)
        if n_components < self.n_components:
            print(f"            [PCA] Capping n_components: {self.n_components} → {n_components}")
            self.n_components = n_components

        _, total = self._boundaries()

        # Compute normalization parameters
        self.view_scale = {key: 1.0 / np.sqrt(self.dim_[key]) for key in self.views} # Inverse square root of dimensionality
        stats           = self._compute_mean_std(paths_ap, paths_ml, paths_3d)  # Mean and std for each view
        for key in self.views:
            self.mean_[key] = stats[key]["mean"]
            self.std_[key]  = stats[key]["std"]
        self.eps_           = self._compute_eps()

        # Build the X matrix (n_samples x total_dim)
        mmap_path = self._compute_X_matrix(paths_ap, paths_ml, paths_3d, tmp_dir=tmp_dir)
        try:
            X = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n_samples, total))

            _, S, Vt = randomized_svd(
                X,
                n_components=n_components,
                n_oversamples=10,
                n_iter=4,
                random_state=self.random_seed,
            )

            # Explained variance
            explained_var  = (S ** 2) / n_samples
            total_variance = np.sum(np.var(X, axis=0, ddof=0))

            self.singular_values_           = S.astype(np.float32)
            self.explained_variance_        = explained_var.astype(np.float32)
            self.explained_variance_ratio_  = (explained_var / total_variance).astype(np.float32)
            self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

            norms = np.linalg.norm(Vt, axis=1, keepdims=True)
            self.components_ = (Vt / norms).astype(np.float32)

            del X
            gc.collect()
        finally:
            time.sleep(0.1)
            os.unlink(mmap_path)

        return self

    def inverse_transform(self, z, n_components=None):
        """Reconstruct flat, denormalised arrays from latent codes Y."""
        k         = n_components or self.n_components
        x_joint   = (z @ self.components_[:k]).squeeze(0)
        starts, _ = self._boundaries()
        return {
            key: self._denormalize(
                x_joint[starts[key]: starts[key] + self.dim_[key]] / self.view_scale[key], key
            )
            for key in self.views
        }
    
    def reconstruct(self, z, n_components=None):
        """Reconstruct and reshape each view to its original spatial shape."""
        flat = self.inverse_transform(z, n_components=n_components)
        return {key: flat[key].reshape(self.original_shape_[key]) for key in self.views}

    def encode_2d(self, ap, ml, n_components=None, reg_strength=0):
        """Recover latent code z from ap/ml SVFs only."""
        starts, _ = self._boundaries()
        k         = n_components or self.n_components

        x_ap = self.view_scale['ap']  * self._normalize(ap,  "ap")
        x_ml = self.view_scale['ml'] * self._normalize(ml, "ml")

        x_obs = np.concatenate([x_ap, x_ml]).astype(np.float64)

        ap_sl = slice(starts["ap"], starts["ap"] + self.dim_["ap"])
        ml_sl = slice(starts["ml"], starts["ml"] + self.dim_["ml"])

        # Restrict joint components to the observed (ap/ml) blocks
        U_obs = np.hstack([
            self.components_[:k, ap_sl],
            self.components_[:k, ml_sl]
        ]).astype(np.float64)

        # Regularization
        ev = self.explained_variance_[:k].astype(np.float64)
        eps = max(1e-8, ev.max() * 1e-6)

        prior_diag = 1.0 / (ev + eps)

        b = U_obs @ x_obs
        UUtT = U_obs @ U_obs.T

        # Solve (U_obs U_obs^T + reg_strength * prior_diag) z = U_obs x_obs
        A = UUtT.copy()
        A.flat[::A.shape[0] + 1] += reg_strength * prior_diag
        chol = cho_factor(A, lower=True)
        z = cho_solve(chol, b)

        return z.reshape(1, -1).astype(np.float32)

    def save(self, path: Path):
        """Save the fitted model to a .npz file."""
        path = Path(path)
        save_dict = {
            'components':                 self.components_,
            'explained_variance_':        self.explained_variance_,
            'explained_variance_ratio_':  self.explained_variance_ratio_,
            'cumulative_variance_ratio_': self.cumulative_variance_ratio_,
            'singular_values_':           self.singular_values_,
            'n_components':               np.array(self.n_components),
            'random_seed':                np.array(-1 if self.random_seed is None else self.random_seed),
        }

        for key in self.views:
            save_dict[f'eps_{key}']            = np.array(self.eps_[key])
            save_dict[f'view_scale{key}']      = np.array(self.view_scale[key])
            save_dict[f'mean_{key}']           = self.mean_[key]
            save_dict[f'std_{key}']            = self.std_[key]
            save_dict[f'original_shape_{key}'] = np.array(self.original_shape_[key])
            save_dict[f'dim_{key}']            = np.array(self.dim_[key])

        np.savez(path, **save_dict)
        print(f"            Saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load a fitted model from a .npz file."""
        d   = np.load(path, allow_pickle=False)
        saved_random_seed = int(d['random_seed']) if 'random_seed' in d else -1
        pca = cls(
            n_components=int(d['n_components']),
            random_seed=None if saved_random_seed == -1 else saved_random_seed,
        )
        pca.components_                = d['components']
        pca.explained_variance_        = d['explained_variance_']
        pca.explained_variance_ratio_  = d['explained_variance_ratio_']
        pca.cumulative_variance_ratio_ = d['cumulative_variance_ratio_']
        pca.singular_values_           = d['singular_values_']
        pca.eps_            = {}
        pca.view_scale = {}

        for key in pca.views:
            pca.mean_[key]           = d[f'mean_{key}']
            pca.std_[key]            = d[f'std_{key}']
            pca.original_shape_[key] = tuple(d[f'original_shape_{key}'].tolist())
            pca.dim_[key]            = int(d[f'dim_{key}'])
            pca.eps_[key]            = float(d[f'eps_{key}'])
            pca.view_scale[key]      = float(d[f'view_scale{key}'])

        return pca