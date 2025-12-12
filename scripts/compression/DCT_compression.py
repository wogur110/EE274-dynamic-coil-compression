import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

# Add parent directory to path (go up 1 level from scripts/ to EE274-dynamic-coil-compression/)
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
    get_poisson_mask,
)
from utils.plot_utils import save_rd_curve
import zlib

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

def dct_compress_decompress(kspace, keep_ratio, quant_bits=8):
    """
    Per-coil 2D DCT compression on k-space data with magnitude thresholding.
    kspace: (N, H, W) complex - k-space data (already undersampled)
    keep_ratio: float (0 to 1), fraction of coefficients to keep (largest magnitude).
    """
    N, H, W = kspace.shape
    num_coeffs = N * H * W
    
    # 1. Separate Real and Imaginary
    real = kspace.real
    imag = kspace.imag
    
    # 2. 2D DCT on k-space (orthonormal DCT type 2)
    dct_real = dctn(real, axes=(-2, -1), norm='ortho')
    dct_imag = dctn(imag, axes=(-2, -1), norm='ortho')
    
    # 3. Thresholding
    # Compute magnitude and keep top coefficients globally across all coils/real/imag
    # Combine coeffs for sorting
    all_coeffs = np.concatenate([dct_real.flatten(), dct_imag.flatten()])
    abs_coeffs = np.abs(all_coeffs)
    
    # Determine threshold
    k = int(len(all_coeffs) * keep_ratio)
    if k == 0:
        threshold = np.inf
    elif k == len(all_coeffs):
        threshold = -1.0
    else:
        partitioned = np.partition(abs_coeffs, -k)
        threshold = partitioned[-k]
        
    # Masking
    mask_real = np.abs(dct_real) >= threshold
    mask_imag = np.abs(dct_imag) >= threshold
    
    # Keep coefficients
    dct_real_kept = dct_real * mask_real
    dct_imag_kept = dct_imag * mask_imag
    
    # 4. Quantize and Encode
    # Combine into complex for quantizer (handles real/imag separation)
    coeffs_kept = dct_real_kept + 1j * dct_imag_kept
    
    # quantize_and_encode separates real/imag and zlib compresses them
    rec_coeffs, bits = quantize_and_encode(coeffs_kept, bits=quant_bits)
    
    # 5. Inverse DCT
    rec_real = idctn(rec_coeffs.real, axes=(-2, -1), norm='ortho')
    rec_imag = idctn(rec_coeffs.imag, axes=(-2, -1), norm='ortho')
    
    rec_kspace = rec_real + 1j * rec_imag
    
    return rec_kspace, bits

def run_dct_experiment():
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "reference", "ref_image.pt")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
        print(f"Reference image not found at {ref_img_path}")
        return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]
    
    # Sweep keep ratios
    ratios = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    quant_bits = 8
    
    print("Starting DCT (Water-filling) compression sweep...")
    
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
        output_dir = os.path.join("results", "compression_result", "dct_compression", f"R{R}")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {'bpp': [], 'psnr': [], 'ssim': [], 'ratio': [], 'R': []}
        
        for r in ratios:
            print(f"\n--- Keep Ratio: {r} ---")
            
            # Compress undersampled k-space
            rec_kspace, bits = dct_compress_decompress(kspace_undersampled, r, quant_bits)
            
            # Convert back to image domain
            rec_imgs = sp.ifft(rec_kspace, axes=(-2, -1))
            
            bpp = bits / num_pixels_total
            
            p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
            
            results['bpp'].append(bpp)
            results['psnr'].append(p)
            results['ssim'].append(s)
            results['ratio'].append(r)
            results['R'].append(R)
            
            print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
            
            # Reconstruct and Plot for this ratio
            from utils.espirit_torch import csm_from_espirit
            from utils.mri_utils import crop_center, get_device
            
            # 1. FFT
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
            
            # Save reconstruction
            rec_path = os.path.join(output_dir, f"rec_img_r{r}.pt")
            torch.save(torch.from_numpy(rec_final), rec_path)
            
            # Plot reconstruction
            plt.figure(figsize=(5, 5))
            plt.imshow(np.flipud(np.abs(rec_final).T), cmap='gray')
            plt.title(f"DCT Ratio={r}, R={R}\nBPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"rec_img_r{r}.png"), bbox_inches='tight')
            plt.close()
        
        # Save results for this acceleration ratio
        torch.save(results, os.path.join(output_dir, "results_dct.pt"))
        save_rd_curve(results, f"DCT R={R}", f"dct_rd_curve_R{R}.png", output_dir=output_dir)
        print(f"Saved results for R={R} to {output_dir}")
    
    print("Done.")

if __name__ == "__main__":
    run_dct_experiment()

