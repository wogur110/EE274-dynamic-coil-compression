import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results():
    input_dir = os.path.join("results", "refer0")
    output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    ref_img_path = os.path.join(input_dir, "ref_image.pt")
    maps_path = os.path.join(input_dir, "sensitivity_maps.pt")
    
    print(f"Loading results from {input_dir}...")
    ref_img = torch.load(ref_img_path)
    maps = torch.load(maps_path)
    
    print(f"Ref image shape: {ref_img.shape}")
    print(f"Maps shape: {maps.shape}")
    
    # Convert to numpy if tensor
    if isinstance(ref_img, torch.Tensor):
        ref_img = ref_img.cpu().numpy()
    if isinstance(maps, torch.Tensor):
        maps = maps.cpu().numpy()

    # Plot Reference Image (Magnitude Only)
    plt.figure(figsize=(6, 5))
    plt.imshow(np.flipud(np.abs(ref_img).T), cmap='gray')
    plt.title("Reference Image (Magnitude)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    save_path_ref = os.path.join(output_dir, "ref_image_plot.png")
    plt.savefig(save_path_ref, bbox_inches='tight', dpi=150)
    print(f"Saved reference image plot to {save_path_ref}")
    plt.close()
    
    # Plot Sensitivity Maps (Magnitude Only, First few coils)
    num_coils_to_plot = 8
    cols = 4
    rows = int(np.ceil(num_coils_to_plot / cols))
    
    plt.figure(figsize=(16, 4 * rows))
    for i in range(num_coils_to_plot):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.flipud(np.abs(maps[i]).T), cmap='gray')
        plt.title(f"Coil {i} Mag")
        plt.axis('off')
        
    save_path_maps = os.path.join(output_dir, "sensitivity_maps_plot.png")
    plt.savefig(save_path_maps, bbox_inches='tight', dpi=150)
    print(f"Saved sensitivity maps plot to {save_path_maps}")
    plt.close()

if __name__ == "__main__":
    plot_results()
