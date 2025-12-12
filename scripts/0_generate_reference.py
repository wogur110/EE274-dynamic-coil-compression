import torch
import numpy as np
import sigpy as sp
import os
import sys

# Add parent directory to path (go up 1 level from scripts/ to EE274-dynamic-coil-compression/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.mri_utils import (
    load_imgs,
    crop_center,
    get_device,
    DATA_DIR,
    normalize_complex_image,
)
from utils.espirit_torch import csm_from_espirit
from utils.plot_reference import plot_results

def generate_reference():
    # Load data
    imgs_np = load_imgs()
    print(f"Data shape: {imgs_np.shape}, dtype: {imgs_np.dtype}")
    
    # Convert to k-space
    print("Converting to k-space...")
    ksp_np = sp.fft(imgs_np, axes=(-2, -1))
    
    device = get_device()
    print(f"Using device: {device}")
    
    ksp = torch.from_numpy(ksp_np).to(device)
    
    # Prepare calibration data
    calib_width = 32
    print(f"Cropping calibration region ({calib_width}x{calib_width})...")
    ksp_cal = crop_center(ksp, calib_width)
    
    # Estimate sensitivity maps
    print("Estimating sensitivity maps using custom ESPIRiT...")
    im_size = ksp.shape[-2:]
    
    maps, eigen_vals = csm_from_espirit(
        ksp_cal,
        im_size=im_size,
        thresh=0.02,
        kernel_width=6,
        crp=None,
        max_iter=30,
        verbose=True
    )
    
    # Move back to CPU/Numpy
    maps = maps.cpu().numpy()
    
    print("Reconstructing reference combined image...")
    weights = np.sum(np.abs(maps)**2, axis=0)
    weights = weights + 1e-16
    
    ref_img = np.sum(imgs_np * np.conj(maps), axis=0) / weights
    ref_img = normalize_complex_image(ref_img)
    
    # Save results
    output_dir = os.path.join("results", "reference")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_img_path = os.path.join(output_dir, "ref_image.pt")
    maps_path = os.path.join(output_dir, "sensitivity_maps.pt")
    
    print(f"Saving reference image to {ref_img_path}")
    torch.save(torch.from_numpy(ref_img), ref_img_path)
    
    print(f"Saving sensitivity maps to {maps_path}")
    torch.save(torch.from_numpy(maps), maps_path)
    
    print("Plotting results...")
    plot_results()
    
    print("Done.")

if __name__ == "__main__":
    generate_reference()
