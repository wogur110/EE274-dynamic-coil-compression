import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals

add_safe_globals([np._core.multiarray.scalar])


def load_results(path):
    """Load results from a .pt file and convert to lists."""
    if not os.path.exists(path):
        return None
    
    data = torch.load(path, weights_only=False)
    # Convert tensors to lists if needed
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return list(x) if x is not None else []
    
    bpp = to_list(data.get('bpp', []))
    psnr = to_list(data.get('psnr', []))
    ssim = to_list(data.get('ssim', []))
    
    if len(bpp) == 0:
        return None
    
    # Sort by bpp for nicer curve
    pairs_psnr = sorted(zip(bpp, psnr))
    pairs_ssim = sorted(zip(bpp, ssim))
    bpp_sorted_psnr, psnr_sorted = zip(*pairs_psnr) if pairs_psnr else ([], [])
    bpp_sorted_ssim, ssim_sorted = zip(*pairs_ssim) if pairs_ssim else ([], [])
    
    return {
        'bpp_psnr': list(bpp_sorted_psnr),
        'psnr': list(psnr_sorted),
        'bpp_ssim': list(bpp_sorted_ssim),
        'ssim': list(ssim_sorted)
    }


def plot_rd_curves_for_R(R, base="results"):
    """Plot RD curves for a specific acceleration ratio R."""
    # Define configs for ALL methods
    all_configs = [
        ("JPEG compression", os.path.join(base, "refer1a_jpeg_compression", f"R{R}", "results_jpeg.pt"), "tab:blue"),
        ("DCT transform + compression", os.path.join(base, "refer1b_dct_compression", f"R{R}", "results_dct.pt"), "tab:orange"),
        ("Coil Decoupling", os.path.join(base, "uniform_coil_compression", f"R{R}", "results_pca.pt"), "tab:red"),
        ("Coil Decoupling + Waterfilling", os.path.join(base, "uniform_coil_compression_waterfilling", f"R{R}", "results_pca_waterfilling.pt"), "tab:green"),
        ("Coil Decoupling + Dynamic masking", os.path.join(base, "dynamic_coil_compression", "optimal", f"R{R}", "results_optimal.pt"), "tab:purple"),
        ("Coil Decoupling + Dynamic masking + Waterfilling", os.path.join(base, "dynamic_coil_compression_waterfilling", "optimal", f"R{R}", "results_optimal_waterfilling.pt"), "tab:brown"),
    ]
    
    # Define configs for COIL COMPRESSION methods only
    coil_configs = [
        ("Coil Decoupling", os.path.join(base, "uniform_coil_compression", f"R{R}", "results_pca.pt"), "tab:red"),
        ("Coil Decoupling + Waterfilling", os.path.join(base, "uniform_coil_compression_waterfilling", f"R{R}", "results_pca_waterfilling.pt"), "tab:green"),
        ("Coil Decoupling + Dynamic masking", os.path.join(base, "dynamic_coil_compression", "optimal", f"R{R}", "results_optimal.pt"), "tab:purple"),
        ("Coil Decoupling + Dynamic masking + Waterfilling", os.path.join(base, "dynamic_coil_compression_waterfilling", "optimal", f"R{R}", "results_optimal_waterfilling.pt"), "tab:brown"),
    ]

    def prepare_plot_data(configs):
        """Helper function to prepare plot data from configs."""
        plot_data = []
        for label, path, color in configs:
            result = load_results(path)
            if result is None:
                print(f"Skipping {label} for R={R}: {path} not found")
                continue
            plot_data.append((label, color, result['bpp_psnr'], result['psnr'], 
                             result['bpp_ssim'], result['ssim']))
        return plot_data

    # Prepare data for both plot types
    all_plot_data = prepare_plot_data(all_configs)
    coil_plot_data = prepare_plot_data(coil_configs)

    if len(all_plot_data) == 0 and len(coil_plot_data) == 0:
        print(f"No data found for R={R}")
        return

    out_dir = base
    os.makedirs(out_dir, exist_ok=True)
    
    # ========== Plot 1: rd_curve (ALL methods) ==========
    if len(all_plot_data) > 0:
        # Create PSNR plot for all methods
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        for label, color, bpp_psnr, psnr, _, _ in all_plot_data:
            ax1.plot(bpp_psnr, psnr, marker='o', color=color, label=label, linewidth=2, markersize=4)
        
        ax1.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=18)
        ax1.set_ylabel("PSNR (dB)", fontsize=18)
        ax1.set_title(f"Rate–Distortion (PSNR) - Acceleration Ratio R={R}", fontsize=21, fontweight='bold')
        # Set xlim based on acceleration ratio
        if R == 16:
            ax1.set_xlim(0, 0.7)
        elif R == 8:
            ax1.set_xlim(0, 1.0)
        elif R == 4:
            ax1.set_xlim(0, 1.5)
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend(loc='best', fontsize=18)
        plt.tight_layout()
        
        out_path_psnr = os.path.join(out_dir, f"rd_curve_R{R}_psnr.png")
        plt.savefig(out_path_psnr, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved PSNR RD curve (all methods) for R={R} to {out_path_psnr}")
        
        # Create SSIM plot for all methods
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        for label, color, _, _, bpp_ssim, ssim in all_plot_data:
            ax2.plot(bpp_ssim, ssim, marker='o', color=color, label=label, linewidth=2, markersize=4)
        
        ax2.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=18)
        ax2.set_ylabel("SSIM", fontsize=18)
        ax2.set_title(f"Rate–Distortion (SSIM) - Acceleration Ratio R={R}", fontsize=21, fontweight='bold')
        # Set xlim based on acceleration ratio
        if R == 16:
            ax2.set_xlim(0, 0.7)
        elif R == 8:
            ax2.set_xlim(0, 1.0)
        elif R == 4:
            ax2.set_xlim(0, 1.5)
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.legend(loc='best', fontsize=18)
        plt.tight_layout()
        
        out_path_ssim = os.path.join(out_dir, f"rd_curve_R{R}_ssim.png")
        plt.savefig(out_path_ssim, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved SSIM RD curve (all methods) for R={R} to {out_path_ssim}")
    
    # ========== Plot 2: coil_compression_rd_curve (COIL COMPRESSION only) ==========
    if len(coil_plot_data) > 0:
        # Create PSNR plot for coil compression methods
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        for label, color, bpp_psnr, psnr, _, _ in coil_plot_data:
            ax3.plot(bpp_psnr, psnr, marker='o', color=color, label=label, linewidth=2, markersize=4)
        
        ax3.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=18)
        ax3.set_ylabel("PSNR (dB)", fontsize=18)
        ax3.set_title(f"Rate–Distortion (PSNR) - Acceleration Ratio R={R}", fontsize=21, fontweight='bold')
        # Set xlim based on acceleration ratio
        if R == 16:
            ax3.set_xlim(0, 0.7)
        elif R == 8:
            ax3.set_xlim(0, 1.0)
        elif R == 4:
            ax3.set_xlim(0, 1.5)
        ax3.grid(True, which="both", ls="--", alpha=0.5)
        ax3.legend(loc='best', fontsize=18)
        plt.tight_layout()
        
        out_path_psnr_coil = os.path.join(out_dir, f"coil_compression_rd_curve_R{R}_psnr.png")
        plt.savefig(out_path_psnr_coil, dpi=200, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved PSNR coil compression RD curve for R={R} to {out_path_psnr_coil}")
        
        # Create SSIM plot for coil compression methods
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        for label, color, _, _, bpp_ssim, ssim in coil_plot_data:
            ax4.plot(bpp_ssim, ssim, marker='o', color=color, label=label, linewidth=2, markersize=4)
        
        ax4.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=18)
        ax4.set_ylabel("SSIM", fontsize=18)
        ax4.set_title(f"Rate–Distortion (SSIM) - Acceleration Ratio R={R}", fontsize=21, fontweight='bold')
        # Set xlim based on acceleration ratio
        if R == 16:
            ax4.set_xlim(0, 0.7)
        elif R == 8:
            ax4.set_xlim(0, 1.0)
        elif R == 4:
            ax4.set_xlim(0, 1.5)
        ax4.grid(True, which="both", ls="--", alpha=0.5)
        ax4.legend(loc='best', fontsize=18)
        plt.tight_layout()
        
        out_path_ssim_coil = os.path.join(out_dir, f"coil_compression_rd_curve_R{R}_ssim.png")
        plt.savefig(out_path_ssim_coil, dpi=200, bbox_inches='tight')
        plt.close(fig4)
        print(f"Saved SSIM coil compression RD curve for R={R} to {out_path_ssim_coil}")


def main():
    """Generate RD curves for each acceleration ratio."""
    base = "results"
    
    # Acceleration ratios
    accel_ratios = [1, 2, 4, 8, 16]
    
    print("Generating RD curves for each acceleration ratio...")
    for R in accel_ratios:
        print(f"\n{'='*60}")
        print(f"Processing R = {R}")
        print(f"{'='*60}")
        plot_rd_curves_for_R(R, base)
    
    print("\nAll RD curves generated!")


if __name__ == "__main__":
    main()
