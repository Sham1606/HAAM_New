"""
HAAM Master Rerun Script - Pipeline Orchestrator
Automates the end-to-end process: Metadata -> Features -> Training -> Adaptation -> Validation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

def run_script(script_name, args=None):
    """Run a python script and wait for completion."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n>>> RUNNING: {script_name}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        # Use subprocess.run to capture output and handle errors
        result = subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
        elapsed = time.time() - start_time
        print(f"\nDONE: {script_name} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {script_name} with exit code {e.returncode}")
        return False

def main():
    print("="*80)
    print("HAAM FULL PIPELINE RERUN TOOL")
    print("="*80)
    print(f"System: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Root: {ROOT_DIR}")
    
    # Check dependencies
    print("\nChecking key dependencies...")
    libs = ['torch', 'whisper', 'librosa', 'sklearn', 'pandas', 'seaborn']
    missing = []
    for lib in libs:
        try:
            __import__(lib.replace('sklearn', 'sklearn'))
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"WARNING: Missing libraries: {', '.join(missing)}")
        print("Please install them before proceeding.")
    else:
        print("All dependencies found.")

    # 1. Clear directories for fresh run? (Optional, reprocess_features has resumption)
    os.makedirs(ROOT_DIR / "models" / "improved", exist_ok=True)
    os.makedirs(ROOT_DIR / "results" / "validation", exist_ok=True)
    os.makedirs(ROOT_DIR / "plots" / "validation", exist_ok=True)
    os.makedirs(ROOT_DIR / "docs", exist_ok=True)

    sequence = [
        ("02a_generate_real_metadata.py", []),
        ("reprocess_features_fixed.py", []),
        ("05_train_improved_model.py", ["--epochs", "20"]), # Moderate for rerun
        ("finetune_iemocap_domain.py", ["--epochs", "15", "--lr", "1e-5"]), 
        ("run_cremad_pipeline.py", []),
        ("validate_cremad_results.py", []),
        ("cremad_error_analysis.py", [])
    ]

    print(f"\nReady to run {len(sequence)} scripts in order.")
    # input("Press Enter to start...")

    for script, args in sequence:
        success = run_script(script, args)
        if not success:
            print(f"\nCRITICAL ERROR: Pipeline halted at {script}")
            sys.exit(1)

    print("\n" + "="*80)
    print("PIPELINE RERUN COMPLETE")
    print("="*80)
    print("Final Reports Generated:")
    print(f"1. Benchmarks: docs/cremad_detailed_report.txt")
    print(f"2. Error Analysis: docs/error_analysis_report.md")
    print(f"3. Visualizations: plots/validation/")
    print("="*80)

if __name__ == "__main__":
    main()
