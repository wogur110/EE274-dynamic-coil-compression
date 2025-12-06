import torch
import numpy as np
import sigpy as sp
import sigpy.mri
import os
import sys
import zlib
from typing import Tuple, Optional

# Constants
# Get project root directory (parent of utils/)
_utils_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_utils_dir)
DATA_DIR = os.path.join(_project_root, "dataset")
IMGS_FILE = os.path.join(DATA_DIR, "imgs.pt")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_device():
    return DEVICE

def get_data_path():
    return IMGS_FILE

def load_imgs(path=None):
    if path is None:
        path = get_data_path()
    print(f"Loading data from {path}...")
    try:
        imgs = torch.load(path, map_location='cpu')
    except FileNotFoundError:
        alt_path = os.path.join(os.path.dirname(path), "for_jae", os.path.basename(path))
        if os.path.exists(alt_path):
             print(f"Found at {alt_path}")
             imgs = torch.load(alt_path, map_location='cpu')
        else:
            raise
            
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.numpy()
    return imgs

def crop_center(img, crop):
    """
    Crops the center `crop` region from the last two dimensions of `img`.
    """
    if isinstance(crop, int):
        crop = (crop, crop)
    
    y, x = img.shape[-2:]
    cropy, cropx = crop
    
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[..., starty:starty+cropy, startx:startx+cropx]

def mse_loss(x, y):
    # Compute MSE on magnitude images
    return np.mean((np.abs(x) - np.abs(y))**2)

def psnr(x, y, max_val=None):
    # Compute PSNR on magnitude images
    if max_val is None:
        # For normalized images (magnitude in [0, 1]), use 1.0 as max_val
        # Otherwise use the actual maximum of the reference
        ref_max = np.max(np.abs(y))
        # If reference is normalized (max close to 1.0), use 1.0 for consistency
        if np.isclose(ref_max, 1.0, atol=1e-3):
            max_val = 1.0
        else:
            max_val = ref_max
    mse = mse_loss(x, y)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def ssim_complex(x, y, max_val=None):
    try:
        from skimage.metrics import structural_similarity as ssim
        if max_val is None:
            max_val = np.max(np.abs(y))
        return ssim(np.abs(x), np.abs(y), data_range=max_val)
    except ImportError:
        print("skimage not found, returning 0 for SSIM")
        return 0

def normalize_complex_image(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes a complex image so that its magnitude spans [0, 1],
    while preserving phase information.
    """
    if img is None:
        return img

    arr = np.array(img, copy=True)
    mag = np.abs(arr)
    min_val = np.min(mag)
    max_val = np.max(mag)
    denom = max(max_val - min_val, eps)
    target_mag = (mag - min_val) / denom

    scale = np.zeros_like(target_mag, dtype=arr.real.dtype)
    mask = mag > eps
    scale[mask] = target_mag[mask] / mag[mask]
    arr *= scale
    return arr

def run_espirit_pipeline(reconstructed_coils_np: np.ndarray, 
                        reference_image_np: np.ndarray,
                        calib_width: int = 32,
                        device=DEVICE,
                        verbose=False) -> Tuple[float, float]:
    """
    Runs ESPIRiT on reconstructed coils, computes combined image, 
    and evaluates against reference.
    """
    # Import locally to avoid circular imports if any, or just proper import
    # Using relative import or absolute package import
    # Since this file is in utils/, we need to import espirit_torch from same dir
    from utils.espirit_torch import csm_from_espirit
    
    # 1. FFT to k-space
    rec_ksp = sp.fft(reconstructed_coils_np, axes=(-2, -1))
    
    # 2. Crop calibration region
    rec_ksp_torch = torch.from_numpy(rec_ksp).to(device)
    ksp_cal = crop_center(rec_ksp_torch, calib_width)
    
    # 3. ESPIRiT
    im_size = reconstructed_coils_np.shape[-2:]
    
    if verbose:
        print("Estimating sensitivity maps...")
        
    maps, _ = csm_from_espirit(
        ksp_cal,
        im_size=im_size,
        thresh=0.02,
        kernel_width=6,
        crp=None,
        max_iter=30,
        verbose=verbose
    )
    
    if isinstance(maps, torch.Tensor):
        maps = maps.cpu().numpy()
        
    # 4. Combine
    weights = np.sum(np.abs(maps)**2, axis=0)
    weights = weights + 1e-16
    rec_final = np.sum(reconstructed_coils_np * np.conj(maps), axis=0) / weights
    rec_final = normalize_complex_image(rec_final)
    
    # 5. Evaluate
    p = psnr(rec_final, reference_image_np)
    s = ssim_complex(rec_final, reference_image_np)
    
    return p, s

def get_poisson_mask(shape, accel, calib=(32, 32), seed=0):
    """
    Generates a Poisson disc mask.
    """
    if len(shape) == 3:
        spatial_shape = shape[-2:]
    else:
        spatial_shape = shape
    
    if accel == 1 :
        mask = np.ones(spatial_shape)
    else :  
        mask = sigpy.mri.poisson(spatial_shape, accel, calib=calib, dtype=np.float32, seed=seed)
    
    if len(shape) == 3:
        mask = mask[None, ...]
        
    return mask

