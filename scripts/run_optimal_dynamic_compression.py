import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.mri_utils import (
    load_imgs,
    run_espirit_pipeline,
    DATA_DIR,
    normalize_complex_image,
    get_device,
    crop_center,
    get_poisson_mask,
)
from utils.plot_utils import save_rd_curve
from utils.espirit_torch import csm_from_espirit

# Import the compression function
sys.path.insert(0, os.path.join(project_root, "scripts"))


def load_optimal_hyperparameters(base="results", use_waterfilling=False):
    """Load optimal hyperparameters from the saved file."""
    method_name = "dynamic_coil_compression_waterfilling" if use_waterfilling else "dynamic_coil_compression"
    suffix = "_waterfilling" if use_waterfilling else ""
    optimal_path = os.path.join(base, method_name, "optimal_hyperparameters", f"all_optimal_hyperparameters{suffix}.pt")
    
    if not os.path.exists(optimal_path):
        print(f"Optimal hyperparameters not found at {optimal_path}")
        print("Please run find_optimal_hyperparameters.py first!")
        return None
    
    optimal = torch.load(optimal_path, weights_only=False)
    return optimal


def run_optimal_compression_for_R(R, optimal_configs, imgs, ref_img, quant_bits=11, use_waterfilling=False):
    """
    Run compression using only optimal hyperparameters for a given acceleration ratio.
    
    Args:
        R: acceleration ratio
        optimal_configs: dict with bpp as keys and {'K', 'cut_ratio', ...} as values
        imgs: input images (N, H, W)
        ref_img: reference image
        quant_bits: quantization bits
        use_waterfilling: if True, use dynamic_coil_compression_waterfilling function
    
    Returns:
        results dict
    """
    # Import the appropriate compression function
    if use_waterfilling:
        from dynamic_coil_compression_waterfilling import dynamic_coil_compress_decompress
    else:
        from dynamic_coil_compression import dynamic_coil_compress_decompress
    
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    method_label = "Waterfilling" if use_waterfilling else "Regular"
    print(f"\n{'='*60}")
    print(f"Running Optimal Compression ({method_label}) for R = {R}")
    print(f"{'='*60}")
    
    # Generate Poisson mask with calibration region
    mask = get_poisson_mask((N, H, W), accel=R, calib=(32, 32), seed=0)
    if mask.ndim == 3:
        mask = mask[0]  # (H, W)
    
    # Convert to k-space and apply mask
    kspace = sp.fft(imgs, axes=(-2, -1))  # (N, H, W)
    kspace_undersampled = kspace * mask[None, :, :]  # Apply mask to all coils
    
    # Create output directory for optimal results
    method_name = "dynamic_coil_compression_waterfilling" if use_waterfilling else "dynamic_coil_compression"
    output_dir = os.path.join("results", method_name, "optimal", f"R{R}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'bpp': [], 'psnr': [], 'ssim': [], 'K': [], 'cut_ratio': [], 'R': []}
    
    # Sort by bpp for consistent processing
    bpp_values = sorted(optimal_configs.keys())
    
    print(f"Processing {len(bpp_values)} optimal configurations...")
    
    for bpp_target in bpp_values:
        config = optimal_configs[bpp_target]
        K = config['K']
        cut_ratio = config['cut_ratio']
        
        print(f"\n--- Optimal config: K={K}, cut_ratio={cut_ratio:.2f} (target BPP={bpp_target:.4f}) ---")
        
        # Run compression with optimal hyperparameters
        rec_imgs, bits = dynamic_coil_compress_decompress(
            imgs, K, cut_ratio, quant_bits, kspace_undersampled=kspace_undersampled
        )
        bpp = bits / num_pixels_total
        
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        results['K'].append(K)
        results['cut_ratio'].append(cut_ratio)
        results['R'].append(R)
        
        print(f"Actual BPP: {bpp:.4f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
        # Reconstruct final ESPIRiT image for saving and visualization
        ksp_rec = sp.fft(rec_imgs, axes=(-2, -1))
        ksp_rec_torch = torch.from_numpy(ksp_rec).to(get_device())
        ksp_cal = crop_center(ksp_rec_torch, 32)
        im_size = rec_imgs.shape[-2:]
        maps, _ = csm_from_espirit(
            ksp_cal,
            im_size=im_size,
            thresh=0.02,
            kernel_width=6,
            crp=None,
            max_iter=30,
            verbose=False
        )
        if isinstance(maps, torch.Tensor):
            maps = maps.cpu().numpy()
        weights = np.sum(np.abs(maps)**2, axis=0) + 1e-16
        rec_final = np.sum(rec_imgs * np.conj(maps), axis=0) / weights
        rec_final = normalize_complex_image(rec_final)
        
        # Save individual result as .pt
        rec_path = os.path.join(output_dir, f"rec_K{K}_cut{cut_ratio:.2f}.pt")
        torch.save(torch.from_numpy(rec_final), rec_path)
        
        # Save visualization as PNG
        plt.figure(figsize=(5, 5))
        plt.imshow(np.flipud(np.abs(rec_final).T), cmap='gray')
        plt.title(f"K={K}, cut_ratio={cut_ratio:.2f}, R={R}\nBPP={bpp:.4f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
        plt.axis('off')
        png_path = os.path.join(output_dir, f"rec_K{K}_cut{cut_ratio:.2f}.png")
        plt.savefig(png_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    # Save results
    suffix = "_waterfilling" if use_waterfilling else ""
    torch.save(results, os.path.join(output_dir, f"results_optimal{suffix}.pt"))
    
    # Plot RD curve
    method_title = f"Optimal Dynamic Coil Compression ({method_label}) R={R}"
    save_rd_curve(results, method_title, 
                 f"optimal_rd_curve_R{R}{suffix}.png", output_dir=output_dir)
    
    print(f"\nSaved optimal compression results for R={R} to {output_dir}")
    
    return results


def main():
    """Run compression using optimal hyperparameters."""
    parser = argparse.ArgumentParser(description='Run optimal dynamic coil compression')
    parser.add_argument('--waterfilling', action='store_true',
                       help='Use dynamic_coil_compression_waterfilling instead of regular')
    args = parser.parse_args()
    
    use_waterfilling = args.waterfilling
    
    # Load optimal hyperparameters
    method_label = "Waterfilling" if use_waterfilling else "Regular"
    print(f"Loading optimal hyperparameters ({method_label})...")
    optimal_all = load_optimal_hyperparameters(use_waterfilling=use_waterfilling)
    
    if optimal_all is None:
        return
    
    # Load data
    print("Loading images and reference...")
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
        print("Reference image not found.")
        return
    
    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]
    quant_bits = 10  # default
    
    all_results = {}
    
    # Run compression for each acceleration ratio
    for R in accel_ratios:
        if R not in optimal_all:
            print(f"No optimal hyperparameters found for R={R}, skipping...")
            continue
        
        optimal_configs = optimal_all[R]
        
        if len(optimal_configs) == 0:
            print(f"Empty optimal configurations for R={R}, skipping...")
            continue
        
        results = run_optimal_compression_for_R(R, optimal_configs, imgs, ref_img, quant_bits, 
                                                use_waterfilling=use_waterfilling)
        all_results[R] = results
    
    # Save combined results
    method_name = "dynamic_coil_compression_waterfilling" if use_waterfilling else "dynamic_coil_compression"
    suffix = "_waterfilling" if use_waterfilling else ""
    combined_path = os.path.join("results", method_name, "optimal", f"all_results_optimal{suffix}.pt")
    torch.save(all_results, combined_path)
    print(f"\nSaved all optimal results to {combined_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

