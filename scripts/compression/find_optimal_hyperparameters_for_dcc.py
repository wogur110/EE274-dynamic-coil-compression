import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_results_for_R(R, base="results", use_waterfilling=False):
    """Load results for a specific acceleration ratio R."""
    if use_waterfilling:
        method_dir = os.path.join("compression_result", "dynamic_coil_compression_waterfilling")
        results_file = "results_dynamic_waterfilling.pt"
    else:
        method_dir = os.path.join("compression_result", "dynamic_coil_compression")
        results_file = "results_dynamic.pt"
    
    results_path = os.path.join(base, method_dir, f"R{R}", results_file)
    if not os.path.exists(results_path):
        print(f"Results not found for R={R}: {results_path}")
        return None
    
    data = torch.load(results_path, weights_only=False)
    
    # Convert to lists/numpy arrays
    def to_array(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        if isinstance(x, list):
            return np.array(x)
        return x
    
    return {
        'bpp': to_array(data.get('bpp', [])),
        'psnr': to_array(data.get('psnr', [])),
        'ssim': to_array(data.get('ssim', [])),
        'K': to_array(data.get('K', [])),
        'cut_ratio': to_array(data.get('cut_ratio', [])),
        'R': to_array(data.get('R', []))
    }


def find_optimal_hyperparameters_for_R(R, base="results", bpp_bins=None, ssim_drop_threshold=0.1, use_waterfilling=False):
    """
    Find optimal (K, cut_ratio) for each bpp value that maximizes SSIM.
    
    Args:
        R: acceleration ratio
        base: base results directory
        bpp_bins: if None, use exact bpp values; if provided, bin bpp values
        ssim_drop_threshold: if SSIM range within a bin exceeds this, skip the bin
        use_waterfilling: if True, use dynamic_coil_compression_waterfilling results
    
    Returns:
        dict with optimal hyperparameters for each bpp
    """
    results = load_results_for_R(R, base, use_waterfilling=use_waterfilling)
    if results is None:
        return None
    
    bpp = results['bpp']
    ssim = results['ssim']
    K = results['K']
    cut_ratio = results['cut_ratio']
    
    if len(bpp) == 0:
        return None
    
    # If bpp_bins is provided, bin the bpp values
    if bpp_bins is not None:
        bpp_binned = np.digitize(bpp, bpp_bins)
        unique_bins = np.unique(bpp_binned)
        optimal = {}
        
        for bin_idx in unique_bins:
            mask = bpp_binned == bin_idx
            if np.sum(mask) == 0:
                continue
            
            # Check SSIM range within this bin
            ssim_in_bin = ssim[mask]
            ssim_range = np.max(ssim_in_bin) - np.min(ssim_in_bin)
            
            # Skip if SSIM drops significantly within the bin
            if ssim_range > ssim_drop_threshold:
                print(f"  Skipping bin {bin_idx} for R={R}: SSIM range {ssim_range:.4f} > threshold {ssim_drop_threshold}")
                continue
            
            # Find index with maximum SSIM in this bin
            max_idx = np.argmax(ssim[mask])
            actual_idx = np.where(mask)[0][max_idx]
            
            # Use bin center as representative bpp
            if bin_idx == 0:
                rep_bpp = bpp_bins[0] if len(bpp_bins) > 0 else bpp[actual_idx]
            elif bin_idx >= len(bpp_bins):
                rep_bpp = bpp_bins[-1] if len(bpp_bins) > 0 else bpp[actual_idx]
            else:
                rep_bpp = (bpp_bins[bin_idx - 1] + bpp_bins[bin_idx]) / 2
            
            optimal[rep_bpp] = {
                'K': int(K[actual_idx]),
                'cut_ratio': float(cut_ratio[actual_idx]),
                'ssim': float(ssim[actual_idx]),
                'psnr': float(results['psnr'][actual_idx]),
                'bpp': float(bpp[actual_idx])
            }
    else:
        # Group by unique bpp values (or very close bpp values)
        # Round bpp to 4 decimal places for grouping
        bpp_rounded = np.round(bpp, decimals=4)
        unique_bpp = np.unique(bpp_rounded)
        optimal = {}
        
        for u_bpp in unique_bpp:
            mask = np.isclose(bpp_rounded, u_bpp, atol=1e-5)
            if np.sum(mask) == 0:
                continue
            
            # Find index with maximum SSIM for this bpp
            max_idx = np.argmax(ssim[mask])
            actual_idx = np.where(mask)[0][max_idx]
            
            optimal[float(u_bpp)] = {
                'K': int(K[actual_idx]),
                'cut_ratio': float(cut_ratio[actual_idx]),
                'ssim': float(ssim[actual_idx]),
                'psnr': float(results['psnr'][actual_idx]),
                'bpp': float(bpp[actual_idx])
            }
    
    return optimal


def plot_optimal_hyperparameters(optimal_dict, R, output_dir, use_waterfilling=False):
    """Plot optimal hyperparameters vs bpp."""
    if optimal_dict is None or len(optimal_dict) == 0:
        print(f"No optimal hyperparameters to plot for R={R}")
        return
    
    # Sort by bpp
    bpp_values = sorted(optimal_dict.keys())
    K_values = [optimal_dict[bpp]['K'] for bpp in bpp_values]
    cut_ratio_values = [optimal_dict[bpp]['cut_ratio'] for bpp in bpp_values]
    ssim_values = [optimal_dict[bpp]['ssim'] for bpp in bpp_values]
    psnr_values = [optimal_dict[bpp]['psnr'] for bpp in bpp_values]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    method_label = "Waterfilling" if use_waterfilling else "Regular"
    
    # Plot 1: Optimal K vs bpp
    ax1 = axes[0, 0]
    ax1.plot(bpp_values, K_values, 'o-', linewidth=2, markersize=6, color='tab:blue')
    ax1.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax1.set_ylabel("Optimal K (number of virtual coils)", fontsize=12)
    ax1.set_title(f"Optimal K vs BPP (R={R}, {method_label})", fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 2: Optimal cut_ratio vs bpp
    ax2 = axes[0, 1]
    ax2.plot(bpp_values, cut_ratio_values, 'o-', linewidth=2, markersize=6, color='tab:orange')
    ax2.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax2.set_ylabel("Optimal cut_ratio", fontsize=12)
    ax2.set_title(f"Optimal cut_ratio vs BPP (R={R}, {method_label})", fontsize=14, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 3: SSIM vs bpp (with optimal hyperparameters)
    ax3 = axes[1, 0]
    ax3.plot(bpp_values, ssim_values, 'o-', linewidth=2, markersize=6, color='tab:green')
    ax3.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax3.set_ylabel("SSIM (optimal)", fontsize=12)
    ax3.set_title(f"Optimal SSIM vs BPP (R={R}, {method_label})", fontsize=14, fontweight='bold')
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot 4: PSNR vs bpp (with optimal hyperparameters)
    ax4 = axes[1, 1]
    ax4.plot(bpp_values, psnr_values, 'o-', linewidth=2, markersize=6, color='tab:red')
    ax4.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax4.set_ylabel("PSNR (optimal, dB)", fontsize=12)
    ax4.set_title(f"Optimal PSNR vs BPP (R={R}, {method_label})", fontsize=14, fontweight='bold')
    ax4.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_waterfilling" if use_waterfilling else ""
    plot_path = os.path.join(output_dir, f"optimal_hyperparameters_R{R}{suffix}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved optimal hyperparameters plot to {plot_path}")


def save_optimal_hyperparameters_table(optimal_dict, R, output_dir, use_waterfilling=False):
    """Save optimal hyperparameters as a table (CSV-like format)."""
    if optimal_dict is None or len(optimal_dict) == 0:
        return
    
    # Sort by bpp
    bpp_values = sorted(optimal_dict.keys())
    
    # Create table
    suffix = "_waterfilling" if use_waterfilling else ""
    table_path = os.path.join(output_dir, f"optimal_hyperparameters_R{R}{suffix}.txt")
    with open(table_path, 'w') as f:
        f.write(f"Optimal Hyperparameters for Acceleration Ratio R={R}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'BPP':<12} {'K':<8} {'cut_ratio':<12} {'SSIM':<10} {'PSNR (dB)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for bpp in bpp_values:
            opt = optimal_dict[bpp]
            f.write(f"{bpp:<12.4f} {opt['K']:<8} {opt['cut_ratio']:<12.2f} "
                   f"{opt['ssim']:<10.4f} {opt['psnr']:<12.2f}\n")
    
    print(f"Saved optimal hyperparameters table to {table_path}")


def main():
    """Find and save optimal hyperparameters for each acceleration ratio."""
    parser = argparse.ArgumentParser(description='Find optimal hyperparameters for dynamic coil compression')
    parser.add_argument('--waterfilling', action='store_true', 
                       help='Use dynamic_coil_compression_waterfilling results instead of regular')
    args = parser.parse_args()
    
    use_waterfilling = args.waterfilling
    
    base = "results"
    accel_ratios = [1, 2, 4, 8, 16]
    # Define unique bpp bin centers from 0.05 to 2.0 (inclusive)
    bpp_bins = np.unique(np.round(np.arange(0, 2.0, 0.05), 4))
    
    # SSIM drop threshold - increased to reduce skipping
    ssim_drop_threshold = 0.3
    
    # Output directory for optimal hyperparameters
    method_subpath = "dynamic_coil_compression_waterfilling" if use_waterfilling else "dynamic_coil_compression"
    output_dir = os.path.join(base, "compression_result", method_subpath, "optimal_hyperparameters")
    os.makedirs(output_dir, exist_ok=True)
    
    method_label = "Waterfilling" if use_waterfilling else "Regular"
    print(f"Finding optimal hyperparameters (K, cut_ratio) for each BPP ({method_label})...")
    print(f"SSIM drop threshold: {ssim_drop_threshold}")
    print("=" * 80)
    
    all_optimal = {}
    
    for R in accel_ratios:
        print(f"\nProcessing R = {R}")
        print("-" * 80)
        
        # Find optimal hyperparameters
        optimal = find_optimal_hyperparameters_for_R(R, base, bpp_bins, 
                                                     ssim_drop_threshold=ssim_drop_threshold,
                                                     use_waterfilling=use_waterfilling)
        
        if optimal is None or len(optimal) == 0:
            print(f"No results found for R={R}")
            continue
        
        all_optimal[R] = optimal
        
        # Plot optimal hyperparameters
        plot_optimal_hyperparameters(optimal, R, output_dir, use_waterfilling=use_waterfilling)
        
        # Save table
        save_optimal_hyperparameters_table(optimal, R, output_dir, use_waterfilling=use_waterfilling)
        
        print(f"Found {len(optimal)} optimal configurations for R={R}")
    
    # Save all optimal hyperparameters
    suffix = "_waterfilling" if use_waterfilling else ""
    all_optimal_path = os.path.join(output_dir, f"all_optimal_hyperparameters{suffix}.pt")
    torch.save(all_optimal, all_optimal_path)
    print(f"\nSaved all optimal hyperparameters to {all_optimal_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Summary of Optimal Hyperparameters ({method_label}):")
    print("=" * 80)
    for R in accel_ratios:
        if R in all_optimal and len(all_optimal[R]) > 0:
            opt = all_optimal[R]
            bpp_min = min(opt.keys())
            bpp_max = max(opt.keys())
            print(f"R={R}: {len(opt)} configurations, BPP range: [{bpp_min:.4f}, {bpp_max:.4f}]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

