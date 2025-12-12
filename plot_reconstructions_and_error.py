import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try importing utils
try:
    from utils.mri_utils import ssim_complex, psnr
except ImportError:
    if os.path.exists("utils/mri_utils.py"):
        from utils.mri_utils import ssim_complex, psnr
    else:
        print("Could not import utils.mri_utils")
        sys.exit(1)

def plot_error(ref, rec, method_name, out_path, error_scale, scale_text):
    # Calculate Metrics
    try:
        val_psnr = psnr(rec, ref)
        val_ssim = ssim_complex(rec, ref)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        val_psnr = 0
        val_ssim = 0

    # Magnitude
    ref_abs = np.abs(ref)
    rec_abs = np.abs(rec)
    
    # Ensure they are normalized to [0, 1]
    ref_max = ref_abs.max() if ref_abs.max() > 0 else 1
    rec_max = rec_abs.max() if rec_abs.max() > 0 else 1
    
    ref_disp = ref_abs / ref_max
    rec_disp = rec_abs / rec_max
    
    # Error map
    diff = np.abs(ref_disp - rec_disp)
    error_map = diff * error_scale
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground Truth
    im0 = axes[0].imshow(np.flipud(ref_disp.T), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Reconstructed
    im1 = axes[1].imshow(np.flipud(rec_disp.T), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"{method_name}\nPSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error Map
    im2 = axes[2].imshow(np.flipud(error_map.T), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Error Map ({scale_text})", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def main():
    # Load Reference Image
    ref_path = '/home/wogur110/Stanford/EE274/project_JH/EE274-dynamic-coil-compression/results/reference/ref_image.pt'
    if not os.path.exists(ref_path):
        print(f"Ref not found at {ref_path}")
        return
    ref_img = torch.load(ref_path, map_location='cpu')
    if isinstance(ref_img, torch.Tensor):
        ref_img = ref_img.numpy()

    jobs = [
        {
            "path": '/home/wogur110/Stanford/EE274/project_JH/EE274-dynamic-coil-compression/results/compression_result/dct_compression/R1/rec_img_r0.1.pt',
            "name": "DCT (r0.1)",
            "fname": "error_map_specific_dct_R1.png",
            "scale": 10.0,
            "text": "x10"
        },
        {
            "path": '/home/wogur110/Stanford/EE274/project_JH/EE274-dynamic-coil-compression/results/compression_result/dct_compression/R2/rec_img_r0.1.pt',
            "name": "DCT (r0.1)",
            "fname": "error_map_specific_dct_R2.png",
            "scale": 2.0,
            "text": "x2"
        }
    ]

    for job in jobs:
        path = job["path"]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"Processing {job['name']}...")
        try:
            img = torch.load(path, map_location='cpu')
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            if img.shape != ref_img.shape:
                print(f"Shape mismatch: {img.shape} vs {ref_img.shape}")
                continue
            
            out_path = os.path.join(project_root, "results/plot", job["fname"])
            plot_error(ref_img, img, job["name"], out_path, job["scale"], job["text"])
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    main()
