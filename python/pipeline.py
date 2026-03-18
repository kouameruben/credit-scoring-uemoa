"""
pipeline.py -- Telecom Churn Predictor Pipeline
Author: Kouame Ruben
"""

import subprocess, sys, time, os
from pathlib import Path

STEPS = [
    ("01_prepare_data.py",  "[1/3] Preparing telecom data..."),
    ("02_churn_model.py",   "[2/3] Training churn models..."),
    ("03_segmentation.py",  "[3/3] Segmentation & retention..."),
]

def run_pipeline():
    print("=" * 60)
    print("  TELECOM CHURN PREDICTOR - PIPELINE")
    print("=" * 60)
    start = time.time()
    this_file = Path(__file__).resolve()
    python_dir = this_file.parent
    project_root = python_dir.parent
    os.chdir(project_root)
    print(f"\n  Working directory: {project_root}")
    
    for script, msg in STEPS:
        print(f"\n{msg}")
        script_path = python_dir / script
        if not script_path.exists():
            print(f"  [ERROR] Script not found: {script_path}")
            return False
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True,
            cwd=str(project_root), env=env, encoding="utf-8", errors="replace"
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  [ERROR] FAILED:")
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"     {line}")
            return False
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  [OK] Pipeline complete in {elapsed:.1f}s")
    print(f"  [>>] Launch: streamlit run dashboard/app.py")
    print(f"{'=' * 60}")
    return True

if __name__ == "__main__":
    run_pipeline()
