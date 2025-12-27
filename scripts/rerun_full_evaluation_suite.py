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

def run_script(script_name, args=None, is_module=False):
    """Run a python script or module and wait for completion."""
    if is_module:
        cmd = [sys.executable, "-m", script_name]
    else:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            return False
        cmd = [sys.executable, str(script_path)]
    
    if args:
        cmd.extend(args)
    
    prefix = "MODULE" if is_module else "SCRIPT"
    print(f"\n>>> RUNNING {prefix}: {script_name}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
        elapsed = time.time() - start_time
        print(f"\nDONE: {script_name} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {script_name} with exit code {e.returncode}")
        return False

def check_environment():
    """Verify critical dependencies are installed."""
    print("Checking environment...")
    
    # 1. Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("  [OK] ffmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "!"*80)
        print("CRITICAL ERROR: 'ffmpeg' not found in system PATH.")
        print("Whisper transcription (Stage 3) will FAIL without ffmpeg.")
        print("Please install ffmpeg (e.g. via 'choco install ffmpeg') and restart.")
        print("!"*80 + "\n")
        sys.exit(1) # HARD EXIT to prevent wasted computation time

    # 2. Check for other libs (optional)
    return True

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

    check_environment()
    
    # Create required directories proactively
    os.makedirs(ROOT_DIR / "models" / "improved", exist_ok=True)
    os.makedirs(ROOT_DIR / "results" / "validation", exist_ok=True)
    os.makedirs(ROOT_DIR / "plots" / "validation", exist_ok=True)
    os.makedirs(ROOT_DIR / "docs", exist_ok=True)
    os.makedirs(ROOT_DIR / "logs", exist_ok=True)

    sequence = [
        ("02a_generate_real_metadata.py", [], False),
        ("02b_split_data.py", [], False),
        ("reprocess_features_fixed.py", [], False),
        ("reprocess_acoustic_features_librosa.py", [], False),
        ("upgrade_features_to_20dim.py", [], False),
        ("train_with_20dim_features.py", [], False), # Priority 3 Ablation
        ("05_train_improved_model.py", [
            "--epochs", "25", 
            "--acoustic-dim", "20", 
            "--feature-dir", "data/processed/features_v4_20dim"
        ], False),
        ("finetune_iemocap_domain.py", [
            "--epochs", "15", 
            "--lr", "1e-5", 
            "--acoustic-dim", "20", 
            "--iemocap-data", "data/processed/features_v4_20dim",
            "--pretrained-model", "models/improved/best_model.pth"
        ], False),
        ("generate_domain_adaptation_report.py", [], False),
        ("run_cremad_pipeline.py", [], False),
        ("src.marathon_layer.visualize_trends", [], True), # Module
        ("src.marathon_layer.train_risk_predictor", [], True), # Module
        ("generate_xai_reports.py", ["--limit", "50"], False),
        ("validate_cremad_results.py", [], False),
        ("cremad_error_analysis.py", [], False)
    ]

    print(f"\nReady to run {len(sequence)} stages in order.")
    # input("Press Enter to start...")

    for script, args, is_mod in sequence:
        success = run_script(script, args, is_module=is_mod)
        if not success:
            print(f"\nCRITICAL ERROR: Pipeline halted at {script}")
            sys.exit(1)

    print("\n" + "="*80)
    print("PIPELINE RERUN COMPLETE")
    print("="*80)
    print("Final Reports Generated:")
    print(f"1. Benchmarks:        docs/cremad_detailed_report.txt")
    print(f"2. Error Analysis:    docs/error_analysis_report.md")
    print(f"3. Ablation Study:    docs/mfcc_ablation_report.json")
    print(f"4. Domain Adaptation: docs/domain_adaptation_report.md")
    print(f"5. XAI Reports:       results/xai_reports/")
    print(f"6. Agent Trends:      results/marathon/agent_trends/")
    print(f"7. Risk Predictor:    saved_models/marathon_risk_predictor.pth")
    print("="*80)

if __name__ == "__main__":
    main()
