import matplotlib.pyplot as plt
import os

def save_rd_curve(results, title, filename, output_dir="results"):
    """
    Plots and saves Rate-Distortion curves for PSNR and SSIM.
    results: dict with 'bpp', 'psnr', 'ssim' lists.
    """
    plt.figure(figsize=(12, 5))
    
    # PSNR
    plt.subplot(1, 2, 1)
    plt.plot(results['bpp'], results['psnr'], 'o-')
    plt.title(f'{title}: Rate-Distortion (PSNR)')
    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    # SSIM
    plt.subplot(1, 2, 2)
    plt.plot(results['bpp'], results['ssim'], 'o-')
    plt.title(f'{title}: Rate-Distortion (SSIM)')
    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

