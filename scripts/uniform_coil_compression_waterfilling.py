import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
import zlib

# Add parent directory to path (go up 1 level from scripts/ to EE274-dynamic-coil-compression/)
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


def quantize_and_encode(coeffs, bits=8):
    """
    Uniform quantization and zlib compression.
    This is the same base quantizer as in uniform_coil_compression.py and is
    used as a building block for the per-coil waterfilling strategy.
    """
    real = coeffs.real
    imag = coeffs.imag

    min_r, max_r = real.min(), real.max()
    min_i, max_i = imag.min(), imag.max()

    levels = 2**bits - 1

    def quant_stream(x, mn, mx):
        if mx == mn:
            return np.zeros_like(
                x, dtype=np.uint8 if bits <= 8 else np.uint16
            ), 0
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

    # Dequantize (simulation)
    def dequant(q, mn, mx):
        if mx == mn:
            return np.full(q.shape, mn)
        scale = (mx - mn) / levels
        return q.astype(np.float32) * scale + mn

    rec_real = dequant(q_real, min_r, max_r)
    rec_imag = dequant(q_imag, min_i, max_i)

    return rec_real + 1j * rec_imag, bits_used


def simple_coil_compression(kspace_data, n_virtual_coils, calib_size=32):
    """
    Coil compression using calibration data in k-space.

    kspace_data: shape (N_coils, H, W) - k-space data
    n_virtual_coils: integer, number of virtual coils to keep
    calib_size: size of calibration region (default 32x32)

    Returns:
        compressed_kspace: (n_virtual_coils, H, W)
        compression_matrix: (N_coils, n_virtual_coils)
        singular_values: (n_virtual_coils,)
    """
    N_coils, H, W = kspace_data.shape

    # 1. Extract calibration data (center calib_size x calib_size region)
    calib_h = calib_size
    calib_w = calib_size
    cy, cx = H // 2, W // 2
    sy = cy - calib_h // 2
    sx = cx - calib_w // 2

    calib_data = kspace_data[:, sy:sy + calib_h, sx:sx + calib_w]  # (N_coils, calib_h, calib_w)

    # 2. Reshape calibration data to [Samples, Coils]
    flat_calib = calib_data.reshape(N_coils, -1).T  # (calib_h * calib_w, N_coils)

    # 3. Compute covariance matrix (Coil x Coil)
    covariance = flat_calib.T.conj() @ flat_calib  # (N_coils, N_coils)

    # 4. Eigen decomposition using SVD (more numerically stable)
    U, S, Vh = np.linalg.svd(covariance)

    # 5. Select top K components (columns of U)
    compression_matrix = U[:, :n_virtual_coils]  # (N_coils, n_virtual_coils)
    singular_values = S[:n_virtual_coils].copy()

    # 6. Apply compression to full k-space
    flat_kspace = kspace_data.reshape(N_coils, -1).T  # (H * W, N_coils)
    compressed_flat = flat_kspace @ compression_matrix  # (H * W, n_virtual_coils)
    compressed_kspace = compressed_flat.T.reshape(n_virtual_coils, H, W)

    return compressed_kspace, compression_matrix, singular_values


def waterfilling_quantize_and_encode(compressed_kspace, singular_values, max_bits=10):
    """
    Waterfilling quantization based on singular values using scaling factors.
    
    Similar to EE274_HW4_ImageCompressor.ipynb approach:
    1. Calculate scaling factors per coil dimension based on log(singular_values)
    2. Multiply compressed_kspace by scaling factors (larger scale = more precision)
    3. Quantize all k-space data at once (for better zlib compression)
    4. Dequantize
    5. Divide by scaling factors
    
    Args:
        compressed_kspace: (K_pca, H, W) complex array
        singular_values: (K_pca,) array of singular values (descending)
        max_bits: quantization bits (used uniformly for all data after scaling)
    
    Returns:
        rec_kspace: (K_pca, H, W) reconstructed k-space after quantization
        total_bits: total number of bits used
    """
    K_pca, H, W = compressed_kspace.shape
    
    # Use log of singular values to define relative importance
    s = np.asarray(singular_values).astype(np.float64)
    eps = 1e-12
    s_clipped = np.clip(s, eps, None)
    
    theta = 0.01
    distortions = np.minimum(s_clipped, theta)
    
    optimal_rate = 0.5 * np.log2(s_clipped / distortions)
    
    # Calculate scaling matrix similar to notebook:
    # scaling_matrix = 0.1 * floor(2^optimal_rate), then clip by max_scale / sqrt(eigenvals)
    base_scale = 0.1
    max_scale = 30.0  # Similar to notebook, can be adjusted for MRI data
    
    # Calculate scaling factors per coil
    scaling_matrix = base_scale * np.floor(np.power(2.0, optimal_rate))  # (K_pca,)
    
    scaling_matrix = np.minimum(scaling_matrix / np.sqrt(s_clipped), max_scale)
    
    # Apply scaling factors to compressed_kspace (per coil dimension)
    # Shape: (K_pca, H, W) * (K_pca, 1, 1) -> (K_pca, H, W)
    scaled_kspace = compressed_kspace * scaling_matrix[:, None, None]
    
    # Quantize all scaled k-space data at once (for better zlib compression)
    scaled_kspace_quantized, bits_coeffs = quantize_and_encode(
        scaled_kspace, bits=max_bits
    )
    
    # Divide by scaling factors to recover original scale
    rec_kspace = scaled_kspace_quantized / scaling_matrix[:, None, None]
    
    return rec_kspace, bits_coeffs


def pca_waterfilling_compress_decompress(
    imgs,
    K_pca,
    quant_bits=10,
    calib_size=32,
    kspace_undersampled=None,
):
    """
    PCA-based uniform coil compression with waterfilling quantization.

    The PCA/KLT stage is identical to uniform_coil_compression.py, but
    the quantization is performed per virtual coil with a bit allocation
    proportional to the singular values (energy) of each component.

    Args:
        imgs: (N, H, W) complex - image domain data
        K_pca: number of virtual coils to keep
        quant_bits: maximum bits per coefficient for the most important coil
        calib_size: calibration region size in k-space
        kspace_undersampled: optional (N, H, W) complex undersampled k-space

    Returns:
        imgs_rec: (K_pca, H, W) complex reconstructed coil images
        total_bits: total bits including basis overhead
    """
    if kspace_undersampled is None:
        kspace = sp.fft(imgs, axes=(-2, -1))  # (N, H, W)
    else:
        kspace = kspace_undersampled

    N, H, W = kspace.shape

    # Coil compression using calibration data
    compressed_kspace, compression_matrix, singular_values = simple_coil_compression(
        kspace, K_pca, calib_size=calib_size
    )

    # Per-coil waterfilling quantization
    compressed_kspace_quantized, bits_coeffs = waterfilling_quantize_and_encode(
        compressed_kspace, singular_values, max_bits=quant_bits
    )

    # Overhead for compression matrix (N * K_pca complex floats, 32 bits each)
    bits_basis = N * K_pca * 2 * 4 * 8
    total_bits = bits_coeffs + bits_basis

    # Convert compressed k-space back to image domain
    imgs_rec = sp.ifft(compressed_kspace_quantized, axes=(-2, -1))

    return imgs_rec, total_bits


def run_pca_waterfilling_experiment():
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    print(f"Loading reference from {ref_img_path}...")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except Exception:
        print("Reference image not found.")
        return

    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels

    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]

    # Sweep K (same set as uniform_coil_compression, without full 64)
    Ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48]
    quant_bits = 10  # maximum bits for the most important virtual coil

    print("Starting PCA waterfilling compression sweep...")

    for R in accel_ratios:
        print(f"\n{'=' * 60}")
        print(f"Acceleration Ratio R = {R}")
        print(f"{'=' * 60}")

        # Generate Poisson mask with calibration region
        mask = get_poisson_mask((N, H, W), accel=R, calib=(32, 32), seed=0)
        if mask.ndim == 3:
            mask = mask[0]  # (H, W)

        # Convert to k-space and apply mask
        kspace = sp.fft(imgs, axes=(-2, -1))  # (N, H, W)
        kspace_undersampled = kspace * mask[None, :, :]

        # Output directory for this acceleration ratio
        output_dir = os.path.join(
            "results", "uniform_coil_compression_waterfilling", f"R{R}"
        )
        os.makedirs(output_dir, exist_ok=True)

        results = {"bpp": [], "psnr": [], "ssim": [], "rank": [], "R": []}

        for K in Ks:
            print(f"\n--- K: {K} ---")

            rec_imgs, bits = pca_waterfilling_compress_decompress(
                imgs,
                K,
                quant_bits,
                calib_size=32,
                kspace_undersampled=kspace_undersampled,
            )
            bpp = bits / num_pixels_total

            p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)

            results["bpp"].append(bpp)
            results["psnr"].append(p)
            results["ssim"].append(s)
            results["rank"].append(K)
            results["R"].append(R)

            print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")

            # Reconstruct final ESPIRiT image for saving
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
                verbose=False,
            )
            if isinstance(maps, torch.Tensor):
                maps = maps.cpu().numpy()
            weights = np.sum(np.abs(maps) ** 2, axis=0) + 1e-16
            rec_final = np.sum(rec_imgs * np.conj(maps), axis=0) / weights
            rec_final = normalize_complex_image(rec_final)

            rec_path = os.path.join(output_dir, f"rec_img_rank{K}.pt")
            torch.save(torch.from_numpy(rec_final), rec_path)

            plt.figure(figsize=(5, 5))
            plt.imshow(np.flipud(np.abs(rec_final).T), cmap="gray")
            plt.title(
                f"PCA Waterfilling Rank={K}, R={R}\n"
                f"BPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}"
            )
            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"rec_img_rank{K}.png"),
                bbox_inches="tight",
            )
            plt.close()

        # Save results for this acceleration ratio
        torch.save(results, os.path.join(output_dir, "results_pca_waterfilling.pt"))

        save_rd_curve(
            results,
            f"PCA (Waterfilling) R={R}",
            f"pca_waterfilling_rd_curve_R{R}.png",
            output_dir=output_dir,
        )
        print(f"Saved results for R={R} to {output_dir}")

    print("Done.")


if __name__ == "__main__":
    run_pca_waterfilling_experiment()


