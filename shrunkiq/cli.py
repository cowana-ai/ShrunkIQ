#!/usr/bin/env python3
import argparse
import os
import sys

from dotenv import load_dotenv

from shrunkiq.config import cfg

# Add the project root to sys.path to allow importing shrunkiq
# This is useful for running directly if this script were outside the package,
# but less critical if it's inside an installed package.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Adjusted for shrunkiq/cli.py
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from shrunkiq.qa import PDFEvaluator

    # from shrunkiq.config import cfg # If specific config models are needed for PDFEvaluator
except ImportError as e:
    print(f"Error importing ShrunkIQ modules: {e}")
    print("Please ensure ShrunkIQ is installed correctly (e.g., pip install -e .) and its dependencies are met.")
    sys.exit(1)

def evaluate_pdf_command(args):
    """Handles the 'evaluate' subcommand."""
    if not os.path.exists(args.original_pdf):
        print(f"Error: Original PDF not found at {args.original_pdf}")
        sys.exit(1)
    if not os.path.exists(args.compressed_pdf):
        print(f"Error: Compressed PDF not found at {args.compressed_pdf}")
        sys.exit(1)

    try:
        evaluator = PDFEvaluator(cfg)
    except Exception as e:
        print(f"Error initializing PDFEvaluator: {e}")
        sys.exit(1)

    # 1. Establish Baseline
    print(f"--- Establishing Baseline for: {args.original_pdf} ---")
    try:
        baseline_eval_result = evaluator.establish_baseline(
            pdf_path=args.original_pdf,
            num_questions_per_page=args.num_questions_per_page,
            zoom=args.zoom
        )
    except Exception as e:
        print(f"Error during baseline establishment: {e}")
        sys.exit(1)

    if not baseline_eval_result or not evaluator.baseline_evaluation_result:
        print("Failed to establish baseline or baseline result is empty. Exiting.")
        sys.exit(1)

    print(f"Baseline BERTScore F1 Mean: {baseline_eval_result.average_f1:.4f}")
    print(f"Baseline Exact Match Accuracy: {baseline_eval_result.get_metric_value('exact_match_accuracy'):.4f}")
    print(f"Baseline Answer Rate: {baseline_eval_result.get_metric_value('answer_rate'):.4f}")

    # 2. Evaluate Processed Document
    if not evaluator.ground_truth_for_comparison:
        print("Critical error: Ground truth for comparison was not set after baseline. Exiting.")
        sys.exit(1)

    print(f"\n--- Evaluating Compressed Document: {args.compressed_pdf} ---")
    try:
        normalized_and_degradation_metrics = evaluator.evaluate_document_against_baseline(
            pdf_path_to_evaluate=args.compressed_pdf,
            zoom=args.zoom
        )
    except Exception as e:
        print(f"Error during compressed document evaluation: {e}")
        sys.exit(1)

    print("\n--- Normalized and Degradation Metrics ---")
    if normalized_and_degradation_metrics:
        for metric_name, value in normalized_and_degradation_metrics.items():
            if value is None or isinstance(value, str) and "undefined" in value:
                print(f"{metric_name}: {value}")
            elif isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")
    else:
        print("No normalized metrics returned.")

def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="ShrunkIQ CLI for evaluating PDF document quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Path to .env file for loading environment variables (e.g., API keys). Applies to all commands."
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Evaluate subcommand parser
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a compressed/processed PDF against an original.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    evaluate_parser.add_argument(
        "original_pdf",
        type=str,
        help="Path to the original PDF file."
    )
    evaluate_parser.add_argument(
        "--compressed_pdf", # Changed from positional to an option
        "-c",
        type=str,
        required=True, # Make it required as it's essential for evaluation
        help="Path to the compressed/processed PDF file."
    )
    evaluate_parser.add_argument(
        "--num_questions_per_page",
        "-n",
        type=int,
        default=3,
        help="Number of questions to generate per page for ground truth."
    )
    evaluate_parser.add_argument(
        "--zoom",
        "-z",
        type=float,
        default=2.0,
        help="Zoom factor for OCR rendering."
    )
    evaluate_parser.set_defaults(func=evaluate_pdf_command) # Set the function to call for this subcommand

    args = parser.parse_args()

    # Load environment variables (once, at the beginning)
    if args.env_path:
        if os.path.exists(args.env_path):
            load_dotenv(dotenv_path=args.env_path)
            print(f"Loaded environment variables from: {args.env_path}")
        else:
            print(f"Warning: Specified .env file not found: {args.env_path}")
    else:
        load_dotenv() # Load from default .env in current dir or parent

    # Call the function associated with the chosen subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # This should not happen if a command is required, but as a fallback:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
