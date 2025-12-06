import torch
import numpy as np
import sigpy as sp
import os
import sys
import io
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path (go up 1 level from scripts/ to EE274-dynamic-coil-compression/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.mri_utils import (
    load_imgs,
    get_device,
    run_espirit_pipeline,
    DATA_DIR,
    normalize_complex_image,
    get_poisson_mask,
)
from utils.plot_utils import save_rd_curve

def tile_images(imgs):
    """
    Tiles a stack of images (N, H, W) into a mosaic (H_mosaic, W_mosaic).
    """
    N, H, W = imgs.shape
    n_side = int(np.ceil(np.sqrt(N)))
    
    mosaic_H = n_side * H
    mosaic_W = n_side * W
    
    mosaic = np.zeros((mosaic_H, mosaic_W), dtype=imgs.dtype)
    
    for i in range(N):
        r = i // n_side
        c = i % n_side
        mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = imgs[i]
        
    return mosaic

def detile_images(mosaic, N, H, W):
    """
    Detiles a mosaic back into a stack of images (N, H, W).
    """
    n_side = int(np.ceil(np.sqrt(N)))
    imgs = np.zeros((N, H, W), dtype=mosaic.dtype)
    
    for i in range(N):
        r = i // n_side
        c = i % n_side
        imgs[i] = mosaic[r*H:(r+1)*H, c*W:(c+1)*W]
        
    return imgs

def quantize_to_uint(x, min_val, max_val, bits=8):
    """
    Quantize to uint8 or uint16.
    bits: 8 for uint8 (0-255), 16 for uint16 (0-65535)
    """
    x = np.clip(x, min_val, max_val)
    x_norm = (x - min_val) / (max_val - min_val + 1e-10)
    max_level = 2**bits - 1
    return (x_norm * max_level).astype(np.uint8 if bits == 8 else np.uint16)

def dequantize_from_uint(x_quantized, min_val, max_val, bits=8):
    """
    Dequantize from uint8 or uint16.
    bits: 8 for uint8, 16 for uint16
    """
    max_level = 2**bits - 1
    x_norm = x_quantized.astype(np.float32) / max_level
    return x_norm * (max_val - min_val) + min_val

def jpeg_compress_decompress(mosaic, quality, quant_bits=8):
    """
    JPEG compression with optional higher precision quantization.
    
    Args:
        mosaic: Input image array
        quality: JPEG quality (1-100)
        quant_bits: Quantization bit depth (8 or 16). Note: JPEG format only 
                    supports 8-bit, so for quant_bits=16, we quantize to uint16
                    first for precision, then convert to uint8 for JPEG.
    """
    min_val = mosaic.min()
    max_val = mosaic.max()
    
    # Quantize to higher precision first (if quant_bits > 8)
    if quant_bits == 16:
        # Quantize to uint16 for higher precision
        mosaic_uint16 = quantize_to_uint(mosaic, min_val, max_val, bits=16)
        # Convert to uint8 for JPEG (map 0-65535 -> 0-255)
        # Use bit shifting: divide by 256 (right shift by 8 bits) for cleaner mapping
        mosaic_uint8 = (mosaic_uint16 >> 8).astype(np.uint8)
    else:
        # Standard uint8 quantization
        mosaic_uint8 = quantize_to_uint(mosaic, min_val, max_val, bits=8)
    
    img_pil = Image.fromarray(mosaic_uint8, mode='L')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    size_bytes = buffer.tell()
    
    buffer.seek(0)
    img_rec_pil = Image.open(buffer)
    mosaic_rec_uint8 = np.array(img_rec_pil)
    
    # Dequantize: for quant_bits=16, we need to map back through uint16
    if quant_bits == 16:
        # Map uint8 back to uint16 range (left shift by 8 bits), then dequantize
        mosaic_rec_uint16 = (mosaic_rec_uint8.astype(np.uint16) << 8)
        mosaic_rec = dequantize_from_uint(mosaic_rec_uint16, min_val, max_val, bits=16)
    else:
        mosaic_rec = dequantize_from_uint(mosaic_rec_uint8, min_val, max_val, bits=8)
    
    return mosaic_rec, size_bytes * 8

def run_jpeg_experiment(quant_bits=8):
    """
    Run JPEG compression experiment with acceleration ratios.
    
    Args:
        quant_bits: Quantization bit depth (8 or 16). For 16-bit, uses higher
                    precision quantization before converting to 8-bit for JPEG.
    """
    # Load data
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    
    print(f"Loading reference from {ref_img_path}...")
    ref_img = torch.load(ref_img_path)
    if isinstance(ref_img, torch.Tensor):
        ref_img = ref_img.numpy()
        
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]
    
    # Expanded quality range
    qualities = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]
    
    print(f"Starting JPEG compression sweep (quant_bits={quant_bits})...")
    
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
        
        # Reconstruct image from undersampled k-space (zero-filled)
        imgs_undersampled = sp.ifft(kspace_undersampled, axes=(-2, -1))  # (N, H, W)
        
        # Now compress in image domain
        imgs_real = imgs_undersampled.real
        imgs_imag = imgs_undersampled.imag
        
        mosaic_real = tile_images(imgs_real)
        mosaic_imag = tile_images(imgs_imag)
        
        # Create output directory for this acceleration ratio
        output_dir = os.path.join("results", "refer1a_jpeg_compression", f"R{R}")
        os.makedirs(output_dir, exist_ok=True)
        
        results = {'bpp': [], 'psnr': [], 'ssim': [], 'quality': [], 'quant_bits': [], 'R': []}
        
        for q in tqdm(qualities, desc=f"R={R}"):
            print(f"\n--- Quality: {q} ---")
            
            rec_mosaic_real, bits_real = jpeg_compress_decompress(mosaic_real, q, quant_bits=quant_bits)
            rec_mosaic_imag, bits_imag = jpeg_compress_decompress(mosaic_imag, q, quant_bits=quant_bits)
            
            total_bits = bits_real + bits_imag
            bpp = total_bits / num_pixels_total  # Bits per complex coil pixel
            
            rec_imgs_real = detile_images(rec_mosaic_real, N, H, W)
            rec_imgs_imag = detile_images(rec_mosaic_imag, N, H, W)
            
            rec_imgs = rec_imgs_real + 1j * rec_imgs_imag
            
            p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
            
            results['bpp'].append(bpp)
            results['psnr'].append(p)
            results['ssim'].append(s)
            results['quality'].append(q)
            results['quant_bits'].append(quant_bits)
            results['R'].append(R)
            
            print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
            
            # Reconstruct for plotting/saving
            from utils.espirit_torch import csm_from_espirit
            from utils.mri_utils import crop_center
            
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
            rec_path = os.path.join(output_dir, f"rec_img_q{q}.pt")
            torch.save(torch.from_numpy(rec_final), rec_path)
            
            # Plot reconstruction
            plt.figure(figsize=(5, 5))
            plt.imshow(np.flipud(np.abs(rec_final).T), cmap='gray')
            plt.title(f"JPEG Q={q}, R={R}\nBPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"rec_img_q{q}.png"), bbox_inches='tight')
            plt.close()

        # Save results for this acceleration ratio
        torch.save(results, os.path.join(output_dir, "results_jpeg.pt"))
        
        # Plot RD Curve for this acceleration ratio
        save_rd_curve(results, f"JPEG R={R}", f"jpeg_rd_curve_R{R}.png", output_dir=output_dir)
        print(f"Saved results for R={R} to {output_dir}")
    
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='JPEG compression with configurable quantization bits')
    parser.add_argument('--quant_bits', type=int, default=8, choices=[8, 16],
                        help='Quantization bit depth (8 or 16). Default: 8')
    args = parser.parse_args()
    
    run_jpeg_experiment(quant_bits=args.quant_bits)
