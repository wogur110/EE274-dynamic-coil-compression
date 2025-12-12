import os
import sys
import subprocess

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    try:
        # Split script_name into script path and arguments
        parts = script_name.split()
        cmd = [sys.executable] + parts
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except Exception as e:
        print(f"Failed to launch {script_name}: {e}")

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # 1. Generate Reference
    run_script(os.path.join("scripts", "0_generate_reference.py"))
    
    # 2. JPEG
    run_script(os.path.join("scripts", "compression", "jpeg_compression.py"))
    
    # 3. DCT Uniform Compression
    run_script(os.path.join("scripts", "compression", "DCT_compression.py"))
    
    # 4. Uniform Coil Compression (PCA)
    run_script(os.path.join("scripts", "compression", "uniform_coil_compression.py"))
    run_script(os.path.join("scripts", "compression", "uniform_coil_compression.py") + " --waterfilling")
    
    # 5. Dynamic Coil Compression (regular)
    run_script(os.path.join("scripts", "compression", "dynamic_coil_compression.py"))
    run_script(os.path.join("scripts", "compression", "find_optimal_hyperparameters_for_dcc.py"))
    run_script(os.path.join("scripts", "compression", "run_optimal_dynamic_compression.py"))

    # 6. Dynamic Coil Compression (waterfilling)
    run_script(os.path.join("scripts", "compression", "dynamic_coil_compression.py") + " --waterfilling")
    run_script(os.path.join("scripts", "compression", "find_optimal_hyperparameters_for_dcc.py") + " --waterfilling")
    run_script(os.path.join("scripts", "compression", "run_optimal_dynamic_compression.py") + " --waterfilling")
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
