#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FLIGHT TELEMETRY - NEURAL NETWORK PIPELINE ORCHESTRATOR
==============================================================================

Purpose: Execute complete neural network pipeline

This script automates the execution of all scripts with dependency checking,
logging, and error handling.

Usage:
    python 06_orchestrator.py              # interactive mode
    python 06_orchestrator.py --all        # run complete pipeline
    python 06_orchestrator.py --steps 3,4  # run specific steps
    python 06_orchestrator.py --neural     # run only neural network steps

Pipeline Steps:
1. 01_exploratory_analysis.py (EDA - reused from classical project)
2. 02_preprocessing.py (Preprocessing - reused from classical project)
3. 03_train_neural_models.py (Train PyTorch and sklearn models)
4. 04_visualize_neural_network.py (Create visualizations)
5. 05_final_report.py (Generate final report)

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import os
import subprocess
import sys
import time
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Try to import colorama for colored output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    class Fore:
        RED = GREEN = BLUE = YELLOW = CYAN = MAGENTA = RESET = ""

    class Style:
        BRIGHT = RESET_ALL = ""


# Configuration Constants
class Config:
    """Orchestrator configuration parameters."""
    # Pipeline scripts
    SCRIPTS = [
        "01_exploratory_analysis.py",
        "02_preprocessing.py",
        "03_train_neural_models.py",
        "04_visualize_neural_network.py",
        "05_final_report.py"
    ]

    SCRIPT_NAMES = [
        "Exploratory Data Analysis",
        "Data Preprocessing",
        "Neural Network Training (PyTorch)",
        "Neural Network Visualizations",
        "Final Report Generation"
    ]

    # File paths
    LOG_FILE = "neural_execution.log"
    INPUT_DATASET = "inputs/voos_telemetria.csv"
    OUTPUT_DIR = Path("outputs")

    # Output directories
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    ARTIFACTS_DIR = OUTPUT_DIR / 'models'
    PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'

    # Key output files
    FINAL_REPORT = OUTPUT_DIR / 'FINAL_REPORT.md'
    PREPROCESSOR_FILE = ARTIFACTS_DIR / 'preprocessor.pkl'
    TRAINING_HISTORY_FILE = RESULTS_DIR / 'neural_training_history.csv'
    WEIGHTS_COMPARISON_FILE = RESULTS_DIR / 'weights_comparison.csv'

    # Execution settings
    SCRIPT_TIMEOUT = 600  # 10 minutes

    # UI strings
    PROMPT_OPTION = "\nOption: "
    PRESS_ENTER = "\nPress ENTER to continue..."

    # Required libraries
    REQUIRED_LIBRARIES = [
        'pandas', 'numpy', 'sklearn', 'matplotlib',
        'seaborn', 'scipy', 'torch'
    ]


# ==============================================================================
# SECTION 2: LOGGING FUNCTIONS
# ==============================================================================


def log(message: str, level: str = "INFO") -> None:
    """Write message to log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def print_colored(
        message: str,
        color: str = Fore.RESET,
        bright: bool = False) -> None:
    """Print colored message if colorama available."""
    if COLORS_AVAILABLE:
        style = Style.BRIGHT if bright else ""
        print(f"{style}{color}{message}{Style.RESET_ALL}")
    else:
        print(message)


def print_header(title: str) -> None:
    """Print formatted header."""
    print()
    print_colored("=" * 120, Fore.CYAN, bright=True)
    print_colored(title.center(120), Fore.CYAN, bright=True)
    print_colored("=" * 120, Fore.CYAN, bright=True)
    print()


def print_separator() -> None:
    """Print separator line."""
    print_colored("─" * 120, Fore.CYAN)


# ==============================================================================
# SECTION 3: DEPENDENCY CHECKING
# ==============================================================================


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    log("Starting dependency check")

    missing = []

    for lib in Config.REQUIRED_LIBRARIES:
        try:
            if lib == 'sklearn':
                __import__('sklearn')
            elif lib == 'torch':
                __import__('torch')
            else:
                __import__(lib)
            print_colored(f"  ✓ {lib}", Fore.GREEN)
        except ImportError:
            print_colored(f"  ✗ {lib} - NOT FOUND", Fore.RED)
            missing.append(lib)

    if missing:
        print()
        print_colored(
            f"⚠️  Missing libraries: {', '.join(missing)}",
            Fore.YELLOW,
            bright=True)
        print_colored(
            f"Install with: pip install {' '.join(missing)}",
            Fore.YELLOW)
        log(f"Missing libraries: {missing}", "WARNING")
        return False

    print()
    print_colored("✓ All dependencies installed", Fore.GREEN, bright=True)
    log("All dependencies OK")
    return True


def check_pytorch() -> None:
    """Check PyTorch installation and configuration."""
    print_header("CHECKING PYTORCH CONFIGURATION")

    try:
        import torch

        print_colored(f"  ✓ PyTorch version: {torch.__version__}", Fore.GREEN)

        # Check CUDA availability
        if torch.cuda.is_available():
            print_colored(
                f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}",
                Fore.GREEN)
            print_colored(
                f"  ✓ CUDA version: {torch.version.cuda}",
                Fore.GREEN)
        else:
            print_colored("  ℹ CUDA not available - will use CPU", Fore.YELLOW)

        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_colored(f"  ✓ Using device: {device}", Fore.GREEN)

        print()

    except ImportError:
        print_colored("  ✗ PyTorch not installed!", Fore.RED)
        print_colored("\nInstall PyTorch:", Fore.YELLOW)
        print("  CPU only:  pip install torch")
        print("  With CUDA: Visit https://pytorch.org/get-started/locally/")
        sys.exit(1)


def check_input_data() -> bool:
    """Check if input data exists."""
    print_header("CHECKING INPUT DATA")
    log("Checking input data")

    input_file = Path(Config.INPUT_DATASET)

    if input_file.exists():
        print_colored(f"✓ Found input dataset: {input_file}", Fore.GREEN)
        size_mb = input_file.stat().st_size / (1024 * 1024)
        print_colored(f"  File size: {size_mb:.2f} MB", Fore.CYAN)
        log(f"Found input dataset: {input_file}")
        return True

    print_colored(f"✗ Input data not found: {input_file}", Fore.RED)
    log("Input data not found", "ERROR")
    return False


# ==============================================================================
# SECTION 4: OUTPUT VERIFICATION
# ==============================================================================


def verify_outputs(step_index: int) -> None:
    """
    Verify expected outputs for a given step.

    Args:
        step_index: Index of the script step (0-based)
    """
    print("\n  Verifying outputs...", end=" ")

    expected_outputs = get_expected_outputs(step_index)
    missing = []
    found = []

    for filepath, description in expected_outputs:
        if Path(filepath).exists():
            found.append(description)
        else:
            missing.append(description)

    if missing:
        print_colored("⚠️  Some outputs missing:", Fore.YELLOW)
        for desc in missing:
            print_colored(f"    ✗ {desc}", Fore.RED)
        if found:
            print_colored(
                f"  But found {
                    len(found)} other outputs:",
                Fore.GREEN)
            for desc in found[:3]:  # Show first 3
                print_colored(f"    ✓ {desc}", Fore.GREEN)
    else:
        print_colored(f"✓ All {len(found)} expected outputs found", Fore.GREEN)


def get_expected_outputs(step_index: int) -> List[Tuple[Path, str]]:
    """
    Get list of expected output files for each pipeline step.

    Args:
        step_index: Index of the script step (0-based)

    Returns:
        List of (filepath, description) tuples
    """
    outputs = {
        0: [  # 01_exploratory_analysis.py
            (Config.GRAPHICS_DIR / "target_hist.png", "Target histogram"),
            (Config.GRAPHICS_DIR / "target_boxplot.png", "Target boxplot"),
            (Config.GRAPHICS_DIR / "heatmap_correlations.png", "Correlation heatmap"),
            (Config.GRAPHICS_DIR / "box_weather_vs_target.png",
             "Weather vs target boxplot"),
            (Config.RESULTS_DIR / "target_statistics.csv", "Target statistics"),
        ],
        1: [  # 02_preprocessing.py
            (Config.ARTIFACTS_DIR / "preprocessor.pkl", "Preprocessor"),
            (Config.ARTIFACTS_DIR / "X_train.pkl", "X_train data"),
            (Config.ARTIFACTS_DIR / "X_test.pkl", "X_test data"),
            (Config.ARTIFACTS_DIR / "y_train.pkl", "y_train data"),
            (Config.ARTIFACTS_DIR / "y_test.pkl", "y_test data"),
            (Config.RESULTS_DIR / "feature_names_after_oh.csv", "Feature names"),
        ],
        2: [  # 03_train_neural_models.py
            (Config.RESULTS_DIR / "neural_training_history.csv", "Training history"),
            (Config.RESULTS_DIR / "weights_comparison.csv", "Weights comparison"),
            (Config.PREDICTIONS_DIR /
             "preds_pytorch_single_neuron.csv", "PyTorch predictions"),
            (Config.PREDICTIONS_DIR /
             "preds_sklearn_linear.csv", "Sklearn predictions"),
        ],
        3: [  # 04_visualize_neural_network.py
            (Config.GRAPHICS_DIR / "neural_learning_curve.png", "Learning curve"),
            (Config.GRAPHICS_DIR / "neural_weights_comparison.png",
             "Weights comparison plot"),
            (Config.GRAPHICS_DIR /
             "neural_vs_sklearn_comparison.png", "Model comparison plot"),
        ],
        4: [  # 05_final_report.py
            (Config.FINAL_REPORT, "Final report"),
        ],
    }

    return outputs.get(step_index, [])


# ==============================================================================
# SECTION 5: SCRIPT EXECUTION
# ==============================================================================


def execute_script(script_path: str, script_name: str,
                   step_number: int) -> Tuple[bool, float, str, str]:
    """
    Execute a single script and return success status.

    Args:
        script_path: Path to the script
        script_name: Human-readable name of the script
        step_number: Step number in the pipeline

    Returns:
        Tuple of (success, elapsed_time, stdout, stderr)
    """
    print()
    print_separator()
    print_colored(f"▶ [{step_number}/{len(Config.SCRIPTS)}] Executing: {script_name}",
                  Fore.BLUE, bright=True)
    print_separator()

    log(f"Executing {script_path}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=Config.SCRIPT_TIMEOUT
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print_colored(f"✓ Completed: {script_name} ({elapsed:.2f}s)",
                          Fore.GREEN, bright=True)
            log(f"{script_path} completed successfully ({elapsed:.2f}s)", "SUCCESS")
            return True, elapsed, result.stdout, result.stderr
        else:
            print_colored(f"✗ Error: {script_name}", Fore.RED, bright=True)
            log(f"{script_path} failed", "ERROR")
            log(f"Error details: {result.stderr}", "ERROR")
            return False, elapsed, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        elapsed = Config.SCRIPT_TIMEOUT
        error_msg = f"Script exceeded {
            Config.SCRIPT_TIMEOUT // 60} minutes timeout"
        print_colored(
            f"✗ Timeout: {script_name} ({elapsed}s)",
            Fore.RED,
            bright=True)
        log(f"{script_path} timed out", "ERROR")
        return False, elapsed, "", error_msg
    except Exception as e:
        elapsed = 0
        error_msg = str(e)
        print_colored(
            f"✗ Exception: {script_name} - {error_msg}", Fore.RED, bright=True)
        log(f"{script_path} exception: {error_msg}", "ERROR")
        return False, elapsed, "", error_msg


def handle_script_error(script_name: str, stdout: str, stderr: str) -> str:
    """
    Handle script execution error and get user decision.

    Args:
        script_name: Name of the failed script
        stdout: Standard output from the script
        stderr: Standard error from the script

    Returns:
        Action to take: 'continue', 'retry', or 'abort'
    """
    print()
    print_colored("═" * 120, Fore.RED)
    print_colored("ERROR DETAILS", Fore.RED, bright=True)
    print_colored("═" * 120, Fore.RED)

    if stderr:
        # Show last 1000 characters of error output
        error_snippet = stderr[-1000:] if len(stderr) > 1000 else stderr
        print_colored("\nError output:", Fore.RED)
        print(error_snippet)

    if stdout and len(stdout) > 0:
        print_colored("\n(Check log file for full output)", Fore.YELLOW)

    print()
    print_colored("What would you like to do?", Fore.YELLOW, bright=True)
    print("  [1] Continue with next step (ignore this error)")
    print("  [2] Retry this step")
    print("  [3] Abort execution")

    while True:
        choice = input("\nChoice: ").strip()
        if choice == '1':
            log(f"User chose to continue after error in {script_name}")
            return 'continue'
        elif choice == '2':
            log(f"User chose to retry {script_name}")
            return 'retry'
        elif choice == '3':
            log(f"User chose to abort after error in {script_name}")
            return 'abort'
        else:
            print_colored("Invalid choice. Please enter 1, 2, or 3.", Fore.RED)


# ==============================================================================
# SECTION 6: PIPELINE EXECUTION
# ==============================================================================


def run_pipeline(steps_to_run: Optional[List[int]] = None) -> None:
    """
    Run the complete pipeline or specific steps.

    Args:
        steps_to_run: List of step indices to run, or None to run all steps
    """
    if steps_to_run is None:
        steps_to_run = list(range(len(Config.SCRIPTS)))

    print_header("STARTING PIPELINE EXECUTION")
    log("=" * 80)
    log("Pipeline execution started")
    log(f"Steps to run: {[i + 1 for i in steps_to_run]}")

    results = []
    total_time = 0

    for i in steps_to_run:
        script = Config.SCRIPTS[i]
        script_name = Config.SCRIPT_NAMES[i]

        # Check if script exists
        if not Path(script).exists():
            print_colored(f"⚠️  Script not found: {script}", Fore.YELLOW)
            log(f"Script not found: {script}", "WARNING")

            cont = input(
                "Continue without this script? (y/n): ").strip().lower()
            if cont == 'y':
                results.append((script_name, False, 0))
                continue
            else:
                print_colored("Execution aborted", Fore.RED)
                log("Execution aborted due to missing script")
                print_summary(results, total_time, aborted=True)
                sys.exit(1)

        # Execute script
        while True:
            success, elapsed, stdout, stderr = execute_script(
                script, script_name, i + 1
            )
            total_time += elapsed

            # Verify outputs
            if success:
                verify_outputs(i)
                results.append((script_name, True, elapsed))
                break
            else:
                results.append((script_name, False, elapsed))
                action = handle_script_error(script_name, stdout, stderr)

                if action == 'continue':
                    break
                elif action == 'retry':
                    print_colored("\nRetrying...", Fore.YELLOW)
                    results.pop()  # Remove failed result before retry
                else:
                    print_colored("\nExecution aborted", Fore.RED)
                    log("Execution aborted by user")
                    print_summary(results, total_time, aborted=True)
                    sys.exit(1)

    print_summary(results, total_time)
    log("Pipeline execution completed")
    log(f"Total time: {total_time:.2f}s")


def print_summary(results: List[Tuple[str, bool, float]],
                  total_time: float, aborted: bool = False) -> None:
    """
    Print execution summary.

    Args:
        results: List of (script_name, success, elapsed_time) tuples
        total_time: Total execution time in seconds
        aborted: Whether execution was aborted
    """
    print()
    print_header("EXECUTION SUMMARY")

    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes

    if aborted:
        print_colored("⚠️  EXECUTION ABORTED", Fore.YELLOW, bright=True)

    print(f"Scripts executed: {len(results)}/{len(Config.SCRIPTS)}")
    print_colored(f"Successes: {successes}", Fore.GREEN)
    if failures > 0:
        print_colored(f"Failures: {failures}", Fore.RED)

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal time: {minutes}m {seconds}s")

    print("\nDetailed results:")
    for script_name, success, elapsed in results:
        status = "✓" if success else "✗"
        color = Fore.GREEN if success else Fore.RED
        print_colored(f"  {status} {script_name} ({elapsed:.2f}s)", color)

    print("\nOutputs verification:")
    check_all_outputs()

    print(f"\nFull log: {Config.LOG_FILE}")

    if successes == len(Config.SCRIPTS):
        print()
        print_colored(
            "🎉 PIPELINE COMPLETED SUCCESSFULLY!",
            Fore.GREEN,
            bright=True)
        if Config.FINAL_REPORT.exists():
            print_colored(
                f"📄 Final report available at: {Config.FINAL_REPORT}",
                Fore.CYAN)

    print_separator()


def check_all_outputs() -> None:
    """Check all expected outputs from all pipeline steps."""
    # Create comprehensive list of all expected outputs
    all_outputs = [
        (Config.INPUT_DATASET, "Input dataset"),

        # Script 01 outputs
        (Config.GRAPHICS_DIR / "target_hist.png", "Target histogram"),
        (Config.GRAPHICS_DIR / "heatmap_correlations.png", "Correlation heatmap"),
        (Config.RESULTS_DIR / "target_statistics.csv", "Target statistics"),

        # Script 02 outputs
        (Config.ARTIFACTS_DIR / "preprocessor.pkl", "Preprocessor"),
        (Config.ARTIFACTS_DIR / "X_train.pkl", "X_train data"),
        (Config.ARTIFACTS_DIR / "X_test.pkl", "X_test data"),
        (Config.ARTIFACTS_DIR / "y_train.pkl", "y_train data"),
        (Config.ARTIFACTS_DIR / "y_test.pkl", "y_test data"),
        (Config.RESULTS_DIR / "feature_names_after_oh.csv", "Feature names"),

        # Script 03 outputs
        (Config.RESULTS_DIR / "neural_training_history.csv", "Training history"),
        (Config.RESULTS_DIR / "weights_comparison.csv", "Weights comparison"),
        (Config.PREDICTIONS_DIR /
         "preds_pytorch_single_neuron.csv", "PyTorch predictions"),
        (Config.PREDICTIONS_DIR / "preds_sklearn_linear.csv", "Sklearn predictions"),

        # Script 04 outputs
        (Config.GRAPHICS_DIR / "neural_learning_curve.png", "Learning curve"),
        (Config.GRAPHICS_DIR / "neural_weights_comparison.png", "Weights plot"),

        # Script 05 outputs
        (Config.FINAL_REPORT, "Final report")
    ]

    found_count = 0
    missing_count = 0

    for path, description in all_outputs:
        path_obj = Path(path)
        if path_obj.exists():
            print_colored(f"  ✓ {description}", Fore.GREEN)
            found_count += 1
        else:
            print_colored(f"  ✗ {description}", Fore.RED)
            missing_count += 1

    print()
    print_colored(f"Summary: {found_count} found, {missing_count} missing",
                  Fore.CYAN if missing_count == 0 else Fore.YELLOW)


# ==============================================================================
# SECTION 7: MENU AND USER INTERACTION
# ==============================================================================


def show_main_menu() -> str:
    """
    Show interactive main menu.

    Returns:
        str: User's menu selection
    """
    print_header(
        "FLIGHT TELEMETRY - NEURAL NETWORK PIPELINE\nML Pipeline Orchestrator")

    print("Select execution mode:")
    print("  [1] Run complete pipeline (all 5 scripts)")
    print("  [2] Run specific steps (choose which ones)")
    print("  [3] Run from step N onwards")
    print("  [4] Clean outputs and restart")
    print("  [5] View pipeline status")
    print("  [6] Exit")

    return input(Config.PROMPT_OPTION).strip()


def select_specific_steps() -> Optional[List[int]]:
    """
    Allow user to select specific steps.

    Returns:
        Optional[List[int]]: List of selected step indices, or None if invalid
    """
    print_header("SELECT STEPS TO EXECUTE")

    for i, name in enumerate(Config.SCRIPT_NAMES, 1):
        print(f"  [{i}] {name}")

    print("\nEnter step numbers separated by commas (e.g., 1,3,4):")
    selection = input("Steps: ").strip()

    try:
        steps = [int(s.strip()) - 1 for s in selection.split(',')]
        steps = [s for s in steps if 0 <= s < len(Config.SCRIPTS)]

        if not steps:
            print_colored("Invalid selection", Fore.RED)
            return None

        print("\nSelected steps:")
        for i in steps:
            print(f"  • {Config.SCRIPT_NAMES[i]}")

        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            return steps

    except Exception:
        print_colored("Invalid input", Fore.RED)

    return None


def select_starting_step() -> Optional[List[int]]:
    """
    Allow user to select starting step.

    Returns:
        Optional[List[int]]: List of step indices from N onwards, or None if invalid
    """
    print_header("RUN FROM STEP N ONWARDS")

    for i, name in enumerate(Config.SCRIPT_NAMES, 1):
        print(f"  [{i}] {name}")

    print("\nEnter starting step number:")
    selection = input("Start from step: ").strip()

    try:
        start = int(selection) - 1
        if 0 <= start < len(Config.SCRIPTS):
            steps = list(range(start, len(Config.SCRIPTS)))

            print("\nWill execute:")
            for i in steps:
                print(f"  • {Config.SCRIPT_NAMES[i]}")

            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                return steps
        else:
            print_colored("Invalid step number", Fore.RED)

    except Exception:
        print_colored("Invalid input", Fore.RED)

    return None


def view_pipeline_status() -> None:
    """View current pipeline status."""
    print_header("PIPELINE STATUS")

    print("📁 Directory Structure:")
    dirs_to_check = [
        ("inputs/", "Input data"),
        ("outputs/", "All outputs"),
        ("outputs/graphics/", "Graphics"),
        ("outputs/results/", "Results"),
        ("outputs/models/", "Models"),
        ("outputs/predictions/", "Predictions")
    ]

    for dir_path, description in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                num_files = len(list(path.iterdir()))
                print_colored(
                    f"  ✓ {description}: {num_files} items",
                    Fore.GREEN)
            else:
                print_colored(f"  ✓ {description}", Fore.GREEN)
        else:
            print_colored(f"  ✗ {description}: not found", Fore.RED)

    print("\n📊 Key Outputs:")
    check_all_outputs()

    print(f"\n📝 Log file: {Config.LOG_FILE}")
    if Path(Config.LOG_FILE).exists():
        size_kb = Path(Config.LOG_FILE).stat().st_size / 1024
        print_colored(f"  Size: {size_kb:.1f} KB", Fore.CYAN)


def clean_outputs() -> None:
    """Clean all output files and folders."""
    print_header("CLEAN OUTPUTS")

    items_to_clean = [
        Config.OUTPUT_DIR,
        Config.LOG_FILE,
        "execution.log",
        "__pycache__"
    ]

    print("The following items will be deleted:")
    for item in items_to_clean:
        if Path(item).exists():
            print_colored(f"  • {item}", Fore.YELLOW)

    print()
    print_colored(
        "⚠️  WARNING: This action cannot be undone!",
        Fore.RED,
        bright=True)
    confirm = input("ARE YOU SURE? Type 'yes' to confirm: ").strip().lower()

    if confirm == 'yes':
        deleted_count = 0
        for item in items_to_clean:
            path = Path(item)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted_count += 1
                elif path.is_file():
                    path.unlink()
                    deleted_count += 1
            except Exception as e:
                print_colored(f"Error deleting {item}: {e}", Fore.RED)

        print_colored(f"\n✓ Cleanup completed ({deleted_count} items deleted)",
                      Fore.GREEN, bright=True)
        log("Outputs cleaned by user")
    else:
        print_colored("Cleanup cancelled", Fore.YELLOW)


# ==============================================================================
# SECTION 8: COMMAND LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Orchestrator for Flight Telemetry Neural Network Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 06_orchestrator.py              # Interactive mode
  python 06_orchestrator.py --all        # Run complete pipeline
  python 06_orchestrator.py --steps 1,3  # Run EDA and training only
  python 06_orchestrator.py --from 4     # Run from visualizations onwards
  python 06_orchestrator.py --clean      # Clean all outputs
        """)

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (non-interactive)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Run specific steps (comma-separated, e.g., 1,3,4)'
    )

    parser.add_argument(
        '--from',
        type=int,
        dest='from_step',
        help='Run from step N onwards (e.g., --from 4)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean all outputs'
    )

    parser.add_argument(
        '--silent',
        action='store_true',
        help='Silent mode (no user interaction, auto-continue on errors)'
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and input data checks'
    )

    return parser.parse_args()


# ==============================================================================
# SECTION 9: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main orchestrator function.

    Handles both interactive and command-line modes for running the ML pipeline.
    """
    args = parse_arguments()

    # Handle command line arguments
    if args.clean:
        clean_outputs()
        return

    if args.all or args.steps or args.from_step or args.silent:
        # Non-interactive mode
        log("Starting in non-interactive mode")

        if not args.skip_checks:
            if not check_dependencies():
                print_colored(
                    "\n⚠️  Missing dependencies. Install them or use --skip-checks.",
                    Fore.RED)
                sys.exit(1)

        if args.all:
            run_pipeline()
        elif args.steps:
            try:
                steps = [int(s.strip()) - 1 for s in args.steps.split(',')]
                run_pipeline(steps)
            except Exception:
                print_colored("Invalid steps format", Fore.RED)
                sys.exit(1)
        elif args.from_step:
            if 1 <= args.from_step <= len(Config.SCRIPTS):
                steps = list(range(args.from_step - 1, len(Config.SCRIPTS)))
                run_pipeline(steps)
            else:
                print_colored(
                    f"Invalid step number: {args.from_step}", Fore.RED)
                sys.exit(1)
        return

    # Interactive mode
    log("Starting in interactive mode")

    # Check dependencies
    if not check_dependencies():
        print()
        cont = input("Continue anyway? (y/n): ").strip().lower()
        if cont != 'y':
            sys.exit(1)

    # Check input data
    check_input_data()

    # Main menu loop
    while True:
        choice = show_main_menu()

        if choice == '1':
            run_pipeline()
            break
        elif choice == '2':
            steps = select_specific_steps()
            if steps:
                run_pipeline(steps)
            break
        elif choice == '3':
            steps = select_starting_step()
            if steps:
                run_pipeline(steps)
            break
        elif choice == '4':
            clean_outputs()
            input(Config.PRESS_ENTER)
        elif choice == '5':
            view_pipeline_status()
            input(Config.PRESS_ENTER)
        elif choice == '6':
            print_colored("Goodbye! 👋", Fore.CYAN)
            break
        else:
            print_colored("Invalid option", Fore.RED)
            input(Config.PRESS_ENTER)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("\n⚠️  Execution interrupted by user", Fore.YELLOW)
        log("Execution interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n✗ Unexpected error: {e}", Fore.RED)
        log(f"Unexpected error: {e}", "CRITICAL")
        import traceback
        traceback.print_exc()
        sys.exit(1)
