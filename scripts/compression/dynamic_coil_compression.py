import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
import zlib
import argparse

# Add parent directory to path (go up 2 levels from scripts/compression/ to EE274-dynamic-coil-compression/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
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

def quantize_and_encode(coeffs, bits=8):
    """
    Uniform quantization and zlib compression.
    """
    real = coeffs.real
    imag = coeffs.imag
    
    min_r, max_r = real.min(), real.max()
    min_i, max_i = imag.min(), imag.max()
    
    levels = 2**bits - 1
    
    def quant_stream(x, mn, mx):
        if mx == mn:
            return np.zeros_like(x, dtype=np.uint8 if bits<=8 else np.uint16), 0
        scale = levels / (mx - mn)
        q = np.round((x - mn) * scale)
        q = np.clip(q, 0, levels)
        if bits <= 8:
            return q.astype(np.uint8), 0
        elif bits <= 16:
            return q.astype(np.uint16), 0
        return q.astype(np.uint32), 0
        
    q_real, _ = quant_stream(real, min_r, max_r)
    q_imag, _ = quant_stream(imag, min_i, max_i)
    
    b_real = q_real.tobytes()
    b_imag = q_imag.tobytes()
    
    c_real = zlib.compress(b_real)
    c_imag = zlib.compress(b_imag)
    
    # Bits = compressed size * 8 + overhead (min/max floats)
    bits_used = (len(c_real) + len(c_imag)) * 8 + 4 * 32
    
    # Dequantize (Simulation)
    def dequant(q, mn, mx):
        if mx == mn:
            return np.full(q.shape, mn)
        scale = (mx - mn) / levels
        return q.astype(np.float32) * scale + mn
        
    rec_real = dequant(q_real, min_r, max_r)
    rec_imag = dequant(q_imag, min_i, max_i)
    
    return rec_real + 1j * rec_imag, bits_used

def waterfilling_quantize_and_encode(compressed_kspace, singular_values, max_bits=10):
    """
    Waterfilling quantization based on singular values using scaling factors.
    """
    K_pca, H, W = compressed_kspace.shape
    
    # Treat singular_values as eigenvalues (covariance SVD)
    s = np.asarray(singular_values).astype(np.float64)
    eps = 1e-12
    s_clipped = np.clip(s, eps, None)
    
    # Waterfilling-inspired rate allocation (similar to notebook)
    theta = 0.01
    distortions = np.minimum(s_clipped, theta)
    optimal_rate = 0.5 * np.log2(s_clipped / distortions)
    
    # Calculate scaling matrix
    base_scale = 0.1
    max_scale = 30.0  # Can be tuned for MRI data
    
    scaling_matrix = base_scale * np.floor(np.power(2.0, optimal_rate))  # (K_pca,)
    scaling_matrix = np.minimum(scaling_matrix / np.sqrt(s_clipped), max_scale)
    
    # Apply scaling factors to compressed_kspace (per coil dimension)
    scaled_kspace = compressed_kspace * scaling_matrix[:, None, None]
    
    # Quantize all scaled k-space data at once (for better zlib compression)
    scaled_kspace_quantized, bits_coeffs = quantize_and_encode(
        scaled_kspace, bits=max_bits
    )
    
    # Divide by scaling factors to recover original scale
    rec_kspace = scaled_kspace_quantized / scaling_matrix[:, None, None]
    
    return rec_kspace, bits_coeffs

def simple_coil_compression(kspace_data, n_virtual_coils, calib_size=32):
    """
    Coil compression using calibration data in k-space.
    Returns: compressed k-space (n_virtual_coils, H, W), compression_matrix, singular_values
    """
    N_coils, H, W = kspace_data.shape
    
    # 1. Extract calibration data (center 32x32 region)
    calib_h = calib_size
    calib_w = calib_size
    cy, cx = H // 2, W // 2
    sy = cy - calib_h // 2
    sx = cx - calib_w // 2
    
    calib_data = kspace_data[:, sy:sy+calib_h, sx:sx+calib_w]  # (N_coils, calib_h, calib_w)
    
    # 2. Reshape calibration data
    flat_calib = calib_data.reshape(N_coils, -1).T  # (calib_h * calib_w, N_coils)
    
    # 3. Compute Covariance Matrix
    covariance = flat_calib.T.conj() @ flat_calib  # (N_coils, N_coils)
    
    # 4. Eigen decomposition using SVD
    U, S, Vh = np.linalg.svd(covariance)
    
    # 5. Select top K components
    compression_matrix = U[:, :n_virtual_coils]  # (N_coils, n_virtual_coils)
    singular_values = S[:n_virtual_coils]  # (n_virtual_coils,)
    
    # 6. Apply compression to full k-space
    flat_kspace = kspace_data.reshape(N_coils, -1).T  # (H * W, N_coils)
    compressed_flat = flat_kspace @ compression_matrix  # (H * W, n_virtual_coils)
    compressed_kspace = compressed_flat.T.reshape(n_virtual_coils, H, W)
    
    return compressed_kspace, compression_matrix, singular_values

def get_circular_mask(H, W, radius_ratio):
    """
    Create a circular mask for k-space corner cutting.
    """
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    R = np.sqrt(Y**2 + X**2)
    max_r = np.sqrt(cy**2 + cx**2)
    R_norm = R / max_r
    
    # Keep data within radius_ratio of max radius
    mask = R_norm <= radius_ratio
    return mask

def dynamic_coil_compress_decompress(imgs, K_pca, cut_ratio, quant_bits=8, calib_size=32, kspace_undersampled=None, waterfilling=False):
    """
    Dynamic coil compression with corner cutting based on singular values
    and optional waterfilling quantization.
    """
    if kspace_undersampled is None:
        # Convert to k-space
        kspace = sp.fft(imgs, axes=(-2, -1))  # (N, H, W)
    else:
        kspace = kspace_undersampled
    
    N, H, W = kspace.shape
    
    # 1. Coil compression using calibration data
    compressed_kspace, compression_matrix, singular_values = simple_coil_compression(
        kspace, K_pca, calib_size=calib_size
    )
    
    # 3. Normalize singular values using log-scale for dynamic range compression
    s = singular_values.astype(np.float64)
    eps = 1e-12
    s_clipped = np.clip(s, eps, None)
    s_log = np.log(s_clipped)
    s_min = s_log.min()
    s_max = s_log.max()
    if s_max == s_min:
        S_normalized = np.ones_like(s_log)
    else:
        S_normalized = (s_log - s_min) / (s_max - s_min)  # (K_pca,)
    
    # 4. Apply circular masks to each virtual coil based on singular values
    compressed_kspace_masked = np.zeros_like(compressed_kspace)
    for k in range(K_pca):
        # Calculate radius ratio
        radius_ratio = 1.0 - cut_ratio * (1.0 - S_normalized[k])
        radius_ratio = np.clip(radius_ratio, 0.0, 1.0)
        
        # Create circular mask
        mask = get_circular_mask(H, W, radius_ratio)
        
        # Apply mask
        compressed_kspace_masked[k] = compressed_kspace[k] * mask
    
    # 5. Quantize masked compressed k-space
    if waterfilling:
        compressed_kspace_quantized, bits_coeffs = waterfilling_quantize_and_encode(
            compressed_kspace_masked, singular_values, max_bits=quant_bits
        )
    else:
        compressed_kspace_quantized, bits_coeffs = quantize_and_encode(
            compressed_kspace_masked, bits=quant_bits
        )
    
    # 6. Overhead for compression matrix (N * K_pca complex floats, 32 bits each)
    bits_basis = N * K_pca * 2 * 4 * 8
    
    total_bits = bits_coeffs + bits_basis
    
    # 7. Convert compressed k-space back to image domain
    imgs_rec = sp.ifft(compressed_kspace_quantized, axes=(-2, -1))  # (K_pca, H, W)
    
    return imgs_rec, total_bits

def plot_rd_curves_by_cut_ratio(results, R, output_dir, file_prefix="dynamic"):
    """
    Plot RD curves grouped by cut_ratio with different colors for comparison.
    """
    # Group results by cut_ratio
    cut_ratios = sorted(set(results['cut_ratio']))
    
    # Color map for different cut_ratios
    colors = plt.cm.viridis(np.linspace(0, 1, len(cut_ratios)))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot PSNR curves
    for i, cut_ratio in enumerate(cut_ratios):
        # Filter results for this cut_ratio
        indices = [j for j, cr in enumerate(results['cut_ratio']) if cr == cut_ratio]
        bpp_subset = [results['bpp'][j] for j in indices]
        psnr_subset = [results['psnr'][j] for j in indices]
        
        # Sort by bpp for cleaner curve
        pairs = sorted(zip(bpp_subset, psnr_subset))
        if pairs:
            bpp_sorted, psnr_sorted = zip(*pairs)
            ax1.plot(bpp_sorted, psnr_sorted, 'o-', color=colors[i], 
                    label=f'cut_ratio={cut_ratio:.1f}', linewidth=2, markersize=4)
    
    ax1.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title(f"Dynamic Coil Compression R={R}: PSNR vs cut_ratio", fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot SSIM curves
    for i, cut_ratio in enumerate(cut_ratios):
        # Filter results for this cut_ratio
        indices = [j for j, cr in enumerate(results['cut_ratio']) if cr == cut_ratio]
        bpp_subset = [results['bpp'][j] for j in indices]
        ssim_subset = [results['ssim'][j] for j in indices]
        
        # Sort by bpp for cleaner curve
        pairs = sorted(zip(bpp_subset, ssim_subset))
        if pairs:
            bpp_sorted, ssim_sorted = zip(*pairs)
            ax2.plot(bpp_sorted, ssim_sorted, 'o-', color=colors[i], 
                    label=f'cut_ratio={cut_ratio:.1f}', linewidth=2, markersize=4)
    
    ax2.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title(f"Dynamic Coil Compression R={R}: SSIM vs cut_ratio", fontsize=14, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{file_prefix}_rd_curve_by_cut_ratio_R{R}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cut_ratio comparison plot to {plot_path}")

def run_dynamic_experiment(use_waterfilling):
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "reference", "ref_image.pt")
    print(f"Loading reference from {ref_img_path}...")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
         print("Reference image not found.")
         return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]
    
    # Experiment with two hyperparameters: Ks and cut_ratio
    Ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48]
    cut_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quant_bits = 10
    
    method_name = "Waterfilling" if use_waterfilling else "Regular"
    print(f"Starting Dynamic Coil Compression ({method_name}) sweep...")
    print(f"Ks: {Ks}")
    print(f"cut_ratios: {cut_ratios}")
    
    # Determine output names based on method
    if use_waterfilling:
        subdir_name = "dynamic_coil_compression_waterfilling"
        results_filename = "results_dynamic_waterfilling.pt"
        rd_curve_filename = "dynamic_waterfilling_rd_curve"
        plot_prefix = "dynamic_waterfilling"
    else:
        subdir_name = "dynamic_coil_compression"
        results_filename = "results_dynamic.pt"
        rd_curve_filename = "dynamic_rd_curve"
        plot_prefix = "dynamic"

    # Loop over acceleration ratios
    for R in accel_ratios:
        print(f"\n{'='*60}")
        print(f"Acceleration Ratio R = {R}")
        print(f"{'='*60}")
        
        # Generate Poisson mask with calibration region
        mask = get_poisson_mask((N, H, W), accel=R, calib=(32, 32), seed=0)
        # mask shape: (1, H, W) or (H, W)
        if mask.ndim == 3:
            mask = mask[0]  # (H, W)
        
        # Convert to k-space and apply mask
        kspace = sp.fft(imgs, axes=(-2, -1))  # (N, H, W)
        kspace_undersampled = kspace * mask[None, :, :]  # Apply mask to all coils
        
        # Create output directory for this acceleration ratio
        output_dir = os.path.join("results", "compression_result", subdir_name, f"R{R}")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {'bpp': [], 'psnr': [], 'ssim': [], 'K': [], 'cut_ratio': [], 'R': []}
        
        for K in Ks:
            for cut_ratio in cut_ratios:
                print(f"\n--- K: {K}, cut_ratio: {cut_ratio:.2f} ---")
                
                rec_imgs, bits = dynamic_coil_compress_decompress(
                    imgs, K, cut_ratio, quant_bits, 
                    kspace_undersampled=kspace_undersampled,
                    waterfilling=use_waterfilling
                )
                bpp = bits / num_pixels_total
                
                p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
                
                results['bpp'].append(bpp)
                results['psnr'].append(p)
                results['ssim'].append(s)
                results['K'].append(K)
                results['cut_ratio'].append(cut_ratio)
                results['R'].append(R)
                
                print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
                
                # Save individual result
                rec_path = os.path.join(output_dir, f"rec_K{K}_cut{cut_ratio:.2f}.pt")
                torch.save(torch.from_numpy(rec_imgs), rec_path)
        
        # Save results for this acceleration ratio
        torch.save(results, os.path.join(output_dir, results_filename))
        
        # Plot RD curves grouped by cut_ratio with different colors
        plot_rd_curves_by_cut_ratio(results, R, output_dir, file_prefix=plot_prefix)
        
        # Also save the standard RD curve
        curve_title = f"Dynamic Coil Compression ({method_name}) R={R}"
        save_rd_curve(results, curve_title, f"{rd_curve_filename}_R{R}.png", output_dir=output_dir)
        print(f"Saved results for R={R} to {output_dir}")
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Coil Compression with optional Waterfilling")
    parser.add_argument("--waterfilling", action="store_true", help="Enable waterfilling quantization")
    args = parser.parse_args()
    
    run_dynamic_experiment(args.waterfilling)
