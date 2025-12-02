#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
CLUSTERING PIPELINE ORCHESTRATOR
==============================================================================

Purpose: Execute complete end-to-end clustering analysis pipeline

This script automates the execution of all clustering pipeline scripts with
dependency checking, logging, and error handling.

Usage:
    python 08_orchestrator.py              # interactive mode
    python 08_orchestrator.py --all        # run complete pipeline
    python 08_orchestrator.py --steps 1,3  # run specific steps
    python 08_orchestrator.py --clean      # clean outputs

This script:
1. Runs 01_exploratory_analysis.py (EDA and feature analysis)
2. Runs 02_preprocessing.py (data preparation and scaling)
3. Runs 03_elbow_method.py (optimal k selection)
4. Runs 04_training_evaluation.py (model training and comparison)
5. Runs 05_pca_visualization.py (cluster visualization)
6. Runs 06_dbscan_profile.py (outlier detection and profiling)
7. Runs 07_final_report.py (comprehensive report generation)
8. Provides error handling and progress tracking throughout pipeline

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
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import glob

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
        "03_elbow_method.py",
        "04_training_evaluation.py",
        "05_pca_visualization.py",
        "06_dbscan_profile.py",
        "07_final_report.py"
    ]

    SCRIPT_NAMES = [
        "Exploratory Data Analysis",
        "Data Preprocessing",
        "Elbow Method (Optimal K)",
        "Model Training & Evaluation",
        "PCA Visualization",
        "DBSCAN & Cluster Profiling",
        "Final Report Generation"
    ]

    # File and directory paths
    LOG_FILE = "execution.log"
    INPUT_DATASET = "inputs/voos_telemetria_completa.csv"
    OUTPUT_DIR = Path("outputs")

    # Expected outputs
    DATA_PROCESSED_DIR = OUTPUT_DIR / "data_processed"
    MODELS_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    GRAPHICS_DIR = OUTPUT_DIR / "graphics"

    FINAL_REPORT = OUTPUT_DIR / "FINAL_REPORT.md"

    # User interaction prompts
    PROMPT_OPTION = "\nOption: "
    PRESS_ENTER = "\nPress Enter to continue..."

    # Execution settings
    SCRIPT_TIMEOUT = 600  # 10 minutes

    # Required libraries
    REQUIRED_LIBRARIES = [
        'pandas', 'numpy', 'sklearn', 'matplotlib',
        'seaborn', 'scipy'
    ]

    # Expected outputs after each step - COMPREHENSIVE LIST
    EXPECTED_OUTPUTS: Dict[int, List[str]] = {
        0: [  # 01_exploratory_analysis.py
            str(GRAPHICS_DIR / "histograms.png"),
            str(GRAPHICS_DIR / "boxplots.png"),
            str(GRAPHICS_DIR / "pairplot.png"),
            str(GRAPHICS_DIR / "pairplot.pdf"),
            str(GRAPHICS_DIR / "correlation_heatmap.png"),
            str(DATA_PROCESSED_DIR / "schema_summary.csv"),
            str(OUTPUT_DIR / "eda_notes.md")
        ],
        1: [  # 02_preprocessing.py
            str(DATA_PROCESSED_DIR / "processed_X.npy"),
            str(MODELS_DIR / "standard_scaler.pkl"),
            str(DATA_PROCESSED_DIR / "numeric_features.csv"),
            str(DATA_PROCESSED_DIR / "categorical_features.csv"),
            str(DATA_PROCESSED_DIR / "processed_data_preview.csv"),
            str(OUTPUT_DIR / "preprocessing_notes.md")
        ],
        2: [  # 03_elbow_method.py
            str(DATA_PROCESSED_DIR / "elbow_wcss.csv"),
            str(GRAPHICS_DIR / "elbow_wcss.png"),
            str(GRAPHICS_DIR / "elbow_wcss.pdf"),
            str(OUTPUT_DIR / "elbow_notes.md")
        ],
        3: [  # 04_training_evaluation.py
            # Note: kmeans_k*.pkl and agglomerative_k*.pkl are dynamic
            str(DATA_PROCESSED_DIR / "model_comparison.csv"),
            str(DATA_PROCESSED_DIR / "model_comparison.md"),
            str(MODELS_DIR / "dbscan.pkl"),
            str(GRAPHICS_DIR / "dendrogram.png"),
            str(GRAPHICS_DIR / "dendrogram.pdf"),
            str(OUTPUT_DIR / "train_eval_notes.md")
        ],
        4: [  # 05_pca_visualization.py
            str(GRAPHICS_DIR / "pca_kmeans.png"),
            str(GRAPHICS_DIR / "pca_kmeans.pdf"),
            str(DATA_PROCESSED_DIR / "pca_variance_explained.csv"),
            str(OUTPUT_DIR / "pca_notes.md")
        ],
        5: [  # 06_dbscan_profile.py
            str(GRAPHICS_DIR / "pca_dbscan.png"),
            str(GRAPHICS_DIR / "pca_dbscan.pdf"),
            str(DATA_PROCESSED_DIR / "cluster_profile_means.csv"),
            str(DATA_PROCESSED_DIR / "cluster_profile_means.md"),
            str(OUTPUT_DIR / "profile_notes.md")
        ],
        6: [  # 07_final_report.py
            str(FINAL_REPORT),
            str(RESULTS_DIR / "eda_notes.md"),
            str(RESULTS_DIR / "preprocessing_notes.md"),
            str(RESULTS_DIR / "elbow_notes.md"),
            str(RESULTS_DIR / "train_eval_notes.md"),
            str(RESULTS_DIR / "pca_notes.md"),
            str(RESULTS_DIR / "profile_notes.md")
        ]
    }

    # Optional outputs (won't cause failure if missing)
    OPTIONAL_OUTPUTS: Dict[int, List[str]] = {
        4: [  # 05_pca_visualization.py (agglomerative might not be created)
            str(GRAPHICS_DIR / "pca_agglomerative.png"),
            str(GRAPHICS_DIR / "pca_agglomerative.pdf")
        ]
    }

    # Pattern-based outputs (files that match patterns)
    PATTERN_OUTPUTS: Dict[int, List[str]] = {
        3: [  # 04_training_evaluation.py
            str(MODELS_DIR / "kmeans_k*.pkl"),
            str(MODELS_DIR / "agglomerative_k*.pkl")
        ]
    }


# ==============================================================================
# SECTION 2: LOGGING FUNCTIONS
# ==============================================================================


def log(message: str, level: str = "INFO") -> None:
    """
    Write message to log file with timestamp.

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, CRITICAL, SUCCESS)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def print_colored(
        message: str,
        color: str = Fore.RESET,
        bright: bool = False) -> None:
    """
    Print colored message if colorama available.

    Args:
        message: Message to print
        color: Foreground color
        bright: Whether to use bright/bold style
    """
    if COLORS_AVAILABLE:
        style = Style.BRIGHT if bright else ""
        print(f"{style}{color}{message}{Style.RESET_ALL}")
    else:
        print(message)


def print_header(title: str) -> None:
    """
    Print formatted header.

    Args:
        title: Header title to display
    """
    print()
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print_colored(title.center(80), Fore.CYAN, bright=True)
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print()


def print_separator() -> None:
    """Print separator line."""
    print_colored("─" * 80, Fore.CYAN)


# ==============================================================================
# SECTION 3: DEPENDENCY CHECKING
# ==============================================================================


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    print_header("CHECKING DEPENDENCIES")
    log("Starting dependency check")

    missing = []

    for lib in Config.REQUIRED_LIBRARIES:
        try:
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
            f"Install with: pip install {' '.join(missing)}", Fore.YELLOW)
        log(f"Missing libraries: {missing}", "WARNING")
        return False

    print()
    print_colored("✓ All dependencies installed", Fore.GREEN, bright=True)
    log("All dependencies OK")
    return True


def check_input_data() -> bool:
    """
    Check if input data exists.

    Returns:
        bool: True if data found or user chooses to continue, False to abort
    """
    print_header("CHECKING INPUT DATA")
    log("Checking input data")

    # Check for input file
    input_file = Path(Config.INPUT_DATASET)

    if input_file.exists():
        print_colored(f"✓ Found input dataset: {input_file}", Fore.GREEN)

        # Check file size
        size_mb = input_file.stat().st_size / (1024 * 1024)
        print_colored(f"  File size: {size_mb:.2f} MB", Fore.CYAN)

        log(f"Found input dataset: {input_file}")
        return True

    # Data not found
    print_colored(f"✗ Input data not found: {input_file}", Fore.RED)
    print()
    print_colored("Dataset not found. Options:", Fore.YELLOW)
    print("  [1] Specify path to CSV file")
    print("  [2] Continue anyway (maybe data is already processed)")
    print("  [3] Abort")

    choice = input(Config.PROMPT_OPTION).strip()

    if choice == '1':
        path = input("Enter path to CSV file: ").strip()
        if Path(path).exists():
            print_colored(f"✓ Path found: {path}", Fore.GREEN)
            log(f"User specified data path: {path}")
            return True
        else:
            print_colored("✗ Path not found", Fore.RED)
            return False
    elif choice == '2':
        print_colored(
            "⚠️  Continuing without input verification", Fore.YELLOW)
        log("Continued without input verification", "WARNING")
        return True
    else:
        print_colored("Execution aborted", Fore.RED)
        log("Execution aborted by user")
        sys.exit(0)


# ==============================================================================
# SECTION 4: SCRIPT EXECUTION
# ==============================================================================


def execute_script(script_path: str, script_name: str,
                   step_number: int) -> Tuple[bool, float, str, str]:
    """
    Execute a single script and return success status.

    Args:
        script_path: Path to the script file
        script_name: Human-readable name of the script
        step_number: Current step number

    Returns:
        Tuple[bool, float, str, str]: (success, elapsed_time, stdout, stderr)
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
            env={**os.environ, "PYTHONUTF8": "1"},
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
        print_colored(
            f"✗ Timeout: {script_name} (exceeded {
                Config.SCRIPT_TIMEOUT //
                60} minutes)",
            Fore.RED,
            bright=True)
        log(f"{script_path} timed out", "ERROR")
        return False, Config.SCRIPT_TIMEOUT, "", "Timeout exceeded"
    except Exception as e:
        print_colored(f"✗ Exception: {script_name} - {str(e)}",
                      Fore.RED, bright=True)
        log(f"{script_path} exception: {str(e)}", "ERROR")
        return False, 0, "", str(e)


def handle_script_error(script_name: str, stdout: str, stderr: str) -> str:
    """
    Handle script execution error.

    Args:
        script_name: Name of the failed script
        stdout: Standard output from the script
        stderr: Standard error from the script

    Returns:
        str: User's choice ('continue', 'retry', or 'abort')
    """
    print()
    print_colored("=" * 80, Fore.RED)
    print_colored(f"ERROR EXECUTING: {script_name}", Fore.RED, bright=True)
    print_colored("=" * 80, Fore.RED)

    print("\nWhat would you like to do?")
    print("  [1] Continue to next script (may cause cascade errors)")
    print("  [2] Retry this script")
    print("  [3] Abort execution")
    print("  [4] Debug mode (show full output)")

    choice = input(Config.PROMPT_OPTION).strip()

    if choice == '1':
        log("User chose to continue after error", "WARNING")
        return 'continue'
    elif choice == '2':
        log("User chose to retry script")
        return 'retry'
    elif choice == '4':
        print("\n" + "=" * 80)
        print("STDOUT:")
        print("=" * 80)
        print(stdout if stdout else "(empty)")
        print("\n" + "=" * 80)
        print("STDERR:")
        print("=" * 80)
        print(stderr if stderr else "(empty)")
        print("=" * 80)
        return handle_script_error(script_name, stdout, stderr)
    else:
        log("Execution aborted by user after error")
        return 'abort'


def verify_outputs(step_index: int) -> bool:
    """
    Verify expected outputs were created - ENHANCED VERSION.

    Args:
        step_index: Index of the current step

    Returns:
        bool: True if all expected outputs exist, False otherwise
    """
    if step_index not in Config.EXPECTED_OUTPUTS and \
       step_index not in Config.PATTERN_OUTPUTS and \
       step_index not in Config.OPTIONAL_OUTPUTS:
        return True

    print()
    print_colored("Verifying outputs...", Fore.CYAN)

    all_exist = True

    # Check regular expected outputs
    if step_index in Config.EXPECTED_OUTPUTS:
        for output in Config.EXPECTED_OUTPUTS[step_index]:
            path = Path(output)
            if path.exists():
                if path.is_file():
                    size_kb = path.stat().st_size / 1024
                    print_colored(
                        f"  ✓ {output} ({
                            size_kb:.1f} KB)",
                        Fore.GREEN)
                else:
                    print_colored(f"  ✓ {output}/", Fore.GREEN)
            else:
                print_colored(f"  ✗ {output} - NOT FOUND", Fore.RED)
                log(f"Expected output not found: {output}", "WARNING")
                all_exist = False

    # Check pattern-based outputs (e.g., kmeans_k*.pkl)
    if step_index in Config.PATTERN_OUTPUTS:
        for pattern in Config.PATTERN_OUTPUTS[step_index]:
            matching_files = glob.glob(pattern)
            if matching_files:
                for file in matching_files:
                    path = Path(file)
                    size_kb = path.stat().st_size / 1024
                    print_colored(f"  ✓ {file} ({size_kb:.1f} KB)", Fore.GREEN)
            else:
                print_colored(f"  ✗ {pattern} - NO MATCHES FOUND", Fore.RED)
                log(f"Pattern output not found: {pattern}", "WARNING")
                all_exist = False

    # Check optional outputs (won't affect all_exist)
    if step_index in Config.OPTIONAL_OUTPUTS:
        for output in Config.OPTIONAL_OUTPUTS[step_index]:
            path = Path(output)
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print_colored(
                    f"  ✓ {output} ({
                        size_kb:.1f} KB) [optional]",
                    Fore.CYAN)
            else:
                print_colored(
                    f"  ⚬ {output} - not created [optional]",
                    Fore.YELLOW)

    return all_exist


# ==============================================================================
# SECTION 5: PIPELINE EXECUTION
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
        print_colored(
            f"📄 Final report available at: {
                Config.FINAL_REPORT}",
            Fore.CYAN)

    print_separator()


def check_all_outputs() -> None:
    """Check all expected outputs - COMPREHENSIVE VERSION."""

    # Core outputs to check
    outputs_to_check = [
        # Input
        (Config.INPUT_DATASET, "Input dataset", False),

        # Step 0: EDA
        (Config.GRAPHICS_DIR / "histograms.png", "Histograms", False),
        (Config.GRAPHICS_DIR / "boxplots.png", "Boxplots", False),
        (Config.GRAPHICS_DIR / "pairplot.png", "Pairplot", False),
        (Config.GRAPHICS_DIR / "correlation_heatmap.png",
         "Correlation heatmap", False),
        (Config.DATA_PROCESSED_DIR / "schema_summary.csv", "Schema summary", False),

        # Step 1: Preprocessing
        (Config.DATA_PROCESSED_DIR / "processed_X.npy", "Processed data", False),
        (Config.MODELS_DIR / "standard_scaler.pkl", "Standard scaler", False),
        (Config.DATA_PROCESSED_DIR / "numeric_features.csv",
         "Numeric features list", False),
        (Config.DATA_PROCESSED_DIR / "categorical_features.csv",
         "Categorical features list", False),

        # Step 2: Elbow method
        (Config.DATA_PROCESSED_DIR / "elbow_wcss.csv", "Elbow WCSS data", False),
        (Config.GRAPHICS_DIR / "elbow_wcss.png", "Elbow plot", False),

        # Step 3: Training
        (Config.DATA_PROCESSED_DIR / "model_comparison.csv", "Model comparison", False),
        (Config.MODELS_DIR / "dbscan.pkl", "DBSCAN model", False),
        (Config.GRAPHICS_DIR / "dendrogram.png", "Dendrogram", False),

        # Step 4: PCA Visualization
        (Config.GRAPHICS_DIR / "pca_kmeans.png", "PCA KMeans plot", False),
        (Config.DATA_PROCESSED_DIR /
         "pca_variance_explained.csv", "PCA variance", False),

        # Step 5: DBSCAN profiling
        (Config.GRAPHICS_DIR / "pca_dbscan.png", "PCA DBSCAN plot", False),
        (Config.DATA_PROCESSED_DIR /
         "cluster_profile_means.csv", "Cluster profiles", False),

        # Step 6: Final report
        (Config.FINAL_REPORT, "Final report", False),
        (Config.RESULTS_DIR, "Results directory", True),

        # Optional
        (Config.GRAPHICS_DIR / "pca_agglomerative.png",
         "PCA Agglomerative [optional]", False),
    ]

    # Check pattern-based files
    kmeans_models = glob.glob(str(Config.MODELS_DIR / "kmeans_k*.pkl"))
    agglomerative_models = glob.glob(
        str(Config.MODELS_DIR / "agglomerative_k*.pkl"))

    for path_str, description, is_dir in outputs_to_check:
        path_obj = Path(path_str)
        if path_obj.exists():
            if is_dir:
                count = len(list(path_obj.iterdir())
                            ) if path_obj.is_dir() else 0
                print_colored(f"  ✓ {description} ({count} items)", Fore.GREEN)
            else:
                print_colored(f"  ✓ {description}", Fore.GREEN)
        else:
            if "[optional]" in description:
                print_colored(f"  ⚬ {description}", Fore.YELLOW)
            else:
                print_colored(f"  ✗ {description}", Fore.RED)

    # Check dynamic model files
    if kmeans_models:
        for model in kmeans_models:
            print_colored(f"  ✓ {Path(model).name}", Fore.GREEN)
    else:
        print_colored("  ✗ K-Means models (kmeans_k*.pkl)", Fore.RED)

    if agglomerative_models:
        for model in agglomerative_models:
            print_colored(f"  ✓ {Path(model).name}", Fore.GREEN)
    else:
        print_colored("  ⚬ Agglomerative models [optional]", Fore.YELLOW)


# ==============================================================================
# SECTION 6: MENU AND USER INTERACTION
# ==============================================================================


def show_main_menu() -> str:
    """
    Show interactive main menu.

    Returns:
        str: User's menu selection
    """
    print_header(
        "FLIGHT TELEMETRY CLUSTERING\nML Pipeline Orchestrator")

    print("Select execution mode:")
    print("  [1] Run complete pipeline (all 7 scripts)")
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
        ("outputs/data_processed/", "Processed data"),
        ("outputs/models/", "Model artifacts"),
        ("outputs/results/", "Analysis notes"),
        ("outputs/graphics/", "Visualizations")
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
# SECTION 7: COMMAND LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Orchestrator for Flight Telemetry Clustering ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 08_orchestrator.py              # Interactive mode
  python 08_orchestrator.py --all        # Run complete pipeline
  python 08_orchestrator.py --steps 1,3  # Run EDA and elbow method only
  python 08_orchestrator.py --from 4     # Run from training onwards
  python 08_orchestrator.py --clean      # Clean all outputs
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
# SECTION 8: MAIN FUNCTION
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
                    f"Invalid step number: {
                        args.from_step}", Fore.RED)
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
