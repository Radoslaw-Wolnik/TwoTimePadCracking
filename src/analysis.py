#!/usr/bin/env python3
"""
Two-Time Pad Model Analysis Script
Trains and evaluates language models on Enron emails with proper train/test splits
and generates performance visualizations.
"""

import argparse
import logging
import sys
import random
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataManager
from src.data.enron import EnronDownloader, EnronPreprocessor
from src.model.char_language_model import CharLanguageModel
from src.model.decoder import TwoTimePadDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    byte_accuracy: float
    pair_accuracy: float
    printable_accuracy_1: float
    printable_accuracy_2: float
    word_accuracy_1: float
    word_accuracy_2: float
    total_switches: int
    switch_rate: float
    text_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'byte_accuracy': self.byte_accuracy,
            'pair_accuracy': self.pair_accuracy,
            'printable_accuracy_1': self.printable_accuracy_1,
            'printable_accuracy_2': self.printable_accuracy_2,
            'word_accuracy_1': self.word_accuracy_1,
            'word_accuracy_2': self.word_accuracy_2,
            'total_switches': self.total_switches,
            'switch_rate': self.switch_rate,
            'text_length': self.text_length
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        return cls(**data)


class EmailAnalyzer:
    def __init__(self, res_dir: Path = Path("res")):
        self.res_dir = Path(res_dir)
        self.data_manager = DataManager(self.res_dir)
        self.tenk_dir = self.res_dir / "20k_processed_emails"
        self.results_dir = self.res_dir / "analysis_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def ensure_data_prepared(self, num_emails: int = 20000) -> Path:
        """Ensure emails are downloaded and processed"""
        # Check if raw emails exist
        raw_files = list(self.data_manager.raw_dir.rglob("*"))
        if not raw_files:
            logger.info("Downloading Enron emails...")
            downloader = EnronDownloader(self.data_manager.raw_dir)
            tgz_path = downloader.download()
            extract_dir = self.data_manager.raw_dir / "enron_extracted"
            downloader.extract(tgz_path, extract_dir)
            logger.info(f"Downloaded and extracted to {extract_dir}")
        else:
            logger.info(f"Found {len(raw_files)} raw files, assuming extraction done")

        # Find the extracted directory
        extract_dir = None
        for d in self.data_manager.raw_dir.iterdir():
            if d.is_dir() and "enron" in d.name.lower():
                extract_dir = d
                break

        if not extract_dir:
            raise FileNotFoundError("Could not find extracted Enron emails directory")

        # Process 10k emails if not already done
        self.tenk_dir.mkdir(parents=True, exist_ok=True)

        existing_emails = list(self.tenk_dir.glob("email_*.txt"))
        if len(existing_emails) < num_emails:
            logger.info(f"Processing {num_emails} emails...")
            preprocessor = EnronPreprocessor(extract_dir, self.tenk_dir)
            processed = preprocessor.process(max_emails=num_emails)
            logger.info(f"Processed {processed} emails")
        else:
            logger.info(f"Found {len(existing_emails)} already processed emails")

        return self.tenk_dir

    def load_emails(self, directory: Path, max_emails: int = None) -> List[bytes]:
        """Load emails as bytes for training/testing"""
        email_files = sorted(directory.glob("email_*.txt"))
        if max_emails:
            email_files = email_files[:max_emails]

        emails = []
        for file_path in email_files:
            try:
                text = file_path.read_text(encoding='utf-8', errors='ignore')
                emails.append(text.encode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        logger.info(f"Loaded {len(emails)} emails")
        return emails

    def evaluate_recovery(self, original1: bytes, original2: bytes,
                          recovered1: bytes, recovered2: bytes) -> EvaluationMetrics:
        """Evaluate recovery accuracy with multiple metrics"""
        # Handle empty inputs
        if len(original1) == 0 and len(original2) == 0:
            return EvaluationMetrics(
                byte_accuracy=0.0,
                pair_accuracy=0.0,
                printable_accuracy_1=0.0,
                printable_accuracy_2=0.0,
                word_accuracy_1=0.0,
                word_accuracy_2=0.0,
                total_switches=0,
                switch_rate=0.0,
                text_length=0
            )

        # Convert to numpy arrays
        min_len = min(len(original1), len(original2),
                      len(recovered1), len(recovered2))

        orig1 = np.frombuffer(original1[:min_len], dtype=np.uint8)
        orig2 = np.frombuffer(original2[:min_len], dtype=np.uint8)
        rec1 = np.frombuffer(recovered1[:min_len], dtype=np.uint8)
        rec2 = np.frombuffer(recovered2[:min_len], dtype=np.uint8)

        # Byte accuracy
        correct_bytes_1 = np.sum(orig1 == rec1)
        correct_bytes_2 = np.sum(orig2 == rec2)
        total_bytes = len(orig1) + len(orig2)
        byte_accuracy = (correct_bytes_1 + correct_bytes_2) / total_bytes if total_bytes > 0 else 0.0

        # Pair-wise accuracy
        correct_pairs = 0
        switched_positions = []
        for i in range(min_len):
            if (orig1[i] == rec1[i] and orig2[i] == rec2[i]):
                correct_pairs += 1
            elif (orig1[i] == rec2[i] and orig2[i] == rec1[i]):
                correct_pairs += 1
                switched_positions.append(i)

        pair_accuracy = correct_pairs / min_len if min_len > 0 else 0.0

        # Printable character accuracy
        printable_chars1 = np.sum((orig1 >= 32) & (orig1 <= 126))
        printable_correct1 = np.sum((orig1 == rec1) & (orig1 >= 32) & (orig1 <= 126))
        printable_acc1 = printable_correct1 / printable_chars1 if printable_chars1 > 0 else 0

        printable_chars2 = np.sum((orig2 >= 32) & (orig2 <= 126))
        printable_correct2 = np.sum((orig2 == rec2) & (orig2 >= 32) & (orig2 <= 126))
        printable_acc2 = printable_correct2 / printable_chars2 if printable_chars2 > 0 else 0

        # Word-level accuracy (approximate)
        def bytes_to_words(byte_arr):
            text = byte_arr.tobytes().decode('utf-8', errors='ignore')
            return text.split()

        words1_orig = bytes_to_words(orig1)
        words1_rec = bytes_to_words(rec1)
        words2_orig = bytes_to_words(orig2)
        words2_rec = bytes_to_words(rec2)

        min_words1 = min(len(words1_orig), len(words1_rec))
        word_correct1 = sum(1 for i in range(min_words1) if words1_orig[i] == words1_rec[i])
        word_acc1 = word_correct1 / min_words1 if min_words1 > 0 else 0

        min_words2 = min(len(words2_orig), len(words2_rec))
        word_correct2 = sum(1 for i in range(min_words2) if words2_orig[i] == words2_rec[i])
        word_acc2 = word_correct2 / min_words2 if min_words2 > 0 else 0

        return EvaluationMetrics(
            byte_accuracy=byte_accuracy,
            pair_accuracy=pair_accuracy,
            printable_accuracy_1=printable_acc1,
            printable_accuracy_2=printable_acc2,
            word_accuracy_1=word_acc1,
            word_accuracy_2=word_acc2,
            total_switches=len(switched_positions),
            switch_rate=len(switched_positions) / min_len if min_len > 0 else 0,
            text_length=min_len
        )

    def train_test_split(self, emails: List[bytes], train_ratio: float = 0.8,
                         seed: int = 42) -> Tuple[List[bytes], List[bytes]]:
        """Split emails into training and testing sets"""
        random.seed(seed)
        shuffled = emails.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_set = shuffled[:split_idx]
        test_set = shuffled[split_idx:]

        logger.info(f"Split: {len(train_set)} train, {len(test_set)} test emails")
        return train_set, test_set

    def train_model(self, train_emails: List[bytes], n: int = 7,
                    max_train: int = None) -> CharLanguageModel:
        """Train a character language model on email data"""
        logger.info(f"Training model with n={n} on {len(train_emails)} emails")
        model = CharLanguageModel(n=n)

        train_subset = train_emails[:max_train] if max_train else train_emails

        for i, email in enumerate(train_subset):
            if i % 100 == 0:
                logger.info(f"Training on email {i}/{len(train_subset)}")
            try:
                # Convert bytes to string for training
                text = email.decode('utf-8', errors='ignore')
                model.train(text)
            except Exception as e:
                logger.warning(f"Failed to train on email {i}: {e}")

        logger.info("Training complete")
        return model

    def run_experiment(self, emails: List[bytes], n: int = 7,
                       beam_width: int = 100, num_tests: int = 20,
                       test_length: int = 200, seed: int = 42) -> Dict[str, Any]:
        """Run a single experiment with train/test split"""
        random.seed(seed)

        # Split data
        train_emails, test_emails = self.train_test_split(emails, train_ratio=0.8, seed=seed)

        # Train model
        model = self.train_model(train_emails, n=n)

        # Run tests
        metrics_list = []
        for test_idx in range(num_tests):
            # Skip if not enough test emails
            if len(test_emails) < 2:
                break

            # Randomly select two test emails
            text1, text2 = random.sample(test_emails, 2)

            # Ensure same length
            min_len = min(len(text1), len(text2), test_length)
            if min_len < 50:  # Skip very short texts
                continue

            text1 = text1[:min_len]
            text2 = text2[:min_len]

            # Create XOR stream
            xor_stream = [a ^ b for a, b in zip(text1, text2)]

            # Decode
            decoder = TwoTimePadDecoder(model, model, beam_width=beam_width)
            recovered1, recovered2 = decoder.decode(xor_stream)

            # Evaluate
            metrics = self.evaluate_recovery(text1, text2, recovered1, recovered2)
            metrics_list.append(metrics)

            # Log progress
            if test_idx % 5 == 0:
                logger.info(f"Test {test_idx}: Pair Acc: {metrics.pair_accuracy:.2%}, "
                            f"Byte Acc: {metrics.byte_accuracy:.2%}")

        if not metrics_list:
            raise ValueError("No valid test cases could be run")

        # Calculate averages
        avg_metrics = {}
        for key in metrics_list[0].to_dict().keys():
            values = [getattr(m, key) for m in metrics_list]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
            avg_metrics[f'min_{key}'] = np.min(values)
            avg_metrics[f'max_{key}'] = np.max(values)

        # Count successful tests (pair accuracy > 70%)
        successful = sum(1 for m in metrics_list if m.pair_accuracy > 0.7)
        avg_metrics['success_rate'] = successful / len(metrics_list)
        avg_metrics['num_tests'] = len(metrics_list)

        # Store individual test results
        individual_results = [m.to_dict() for m in metrics_list]

        return {
            'model_config': {'n': n, 'beam_width': beam_width},
            'data_stats': {
                'total_emails': len(emails),
                'train_emails': len(train_emails),
                'test_emails': len(test_emails),
                'avg_test_length': np.mean([m.text_length for m in metrics_list])
            },
            'avg_metrics': avg_metrics,
            'individual_results': individual_results
        }

    def run_cross_validation(self, emails: List[bytes], n: int = 7,
                             beam_width: int = 100, num_runs: int = 5,
                             num_tests_per_run: int = 10) -> Dict[str, Any]:
        """Run multiple experiments with different train/test splits"""
        all_results = []

        for run in range(num_runs):
            logger.info(f"Starting run {run + 1}/{num_runs}")
            seed = 42 + run * 100  # Different seed for each run

            result = self.run_experiment(
                emails=emails,
                n=n,
                beam_width=beam_width,
                num_tests=num_tests_per_run,
                seed=seed
            )

            result['run'] = run + 1
            all_results.append(result)

            # Log summary for this run
            avg_metrics = result['avg_metrics']
            logger.info(f"Run {run + 1} summary: "
                        f"Pair Acc: {avg_metrics['avg_pair_accuracy']:.2%} ± {avg_metrics['std_pair_accuracy']:.3f}, "
                        f"Success Rate: {avg_metrics['success_rate']:.2%}")

        # Calculate overall statistics
        overall_metrics = {}
        for key in all_results[0]['avg_metrics'].keys():
            values = [r['avg_metrics'][key] for r in all_results]
            overall_metrics[f'overall_avg_{key}'] = np.mean(values)
            overall_metrics[f'overall_std_{key}'] = np.std(values)

        return {
            'all_runs': all_results,
            'overall_metrics': overall_metrics,
            'config': {
                'n': n,
                'beam_width': beam_width,
                'num_runs': num_runs,
                'num_tests_per_run': num_tests_per_run,
                'total_emails': len(emails)
            }
        }

    def create_visualizations(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Create visualization plots from experiment results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_files = []

        # Extract data for plotting
        all_runs = results['all_runs']
        pair_accuracies = [r['avg_metrics']['avg_pair_accuracy'] for r in all_runs]
        byte_accuracies = [r['avg_metrics']['avg_byte_accuracy'] for r in all_runs]
        success_rates = [r['avg_metrics']['success_rate'] for r in all_runs]
        run_numbers = list(range(1, len(all_runs) + 1))

        # 1. Accuracy trends across runs
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12))

        # Pair accuracy plot
        axes1[0].plot(run_numbers, pair_accuracies, 'b-o', linewidth=2, markersize=8)
        axes1[0].axhline(y=np.mean(pair_accuracies), color='r', linestyle='--',
                         label=f'Mean: {np.mean(pair_accuracies):.2%}')
        axes1[0].fill_between(run_numbers,
                              [p - np.std(pair_accuracies) for p in pair_accuracies],
                              [p + np.std(pair_accuracies) for p in pair_accuracies],
                              alpha=0.2, color='blue')
        axes1[0].set_xlabel('Run Number')
        axes1[0].set_ylabel('Pair Accuracy')
        axes1[0].set_title('Pair Accuracy Across Runs')
        axes1[0].grid(True, alpha=0.3)
        axes1[0].legend()
        axes1[0].set_ylim(0, 1)

        # Byte accuracy plot
        axes1[1].plot(run_numbers, byte_accuracies, 'g-o', linewidth=2, markersize=8)
        axes1[1].axhline(y=np.mean(byte_accuracies), color='r', linestyle='--',
                         label=f'Mean: {np.mean(byte_accuracies):.2%}')
        axes1[1].fill_between(run_numbers,
                              [b - np.std(byte_accuracies) for b in byte_accuracies],
                              [b + np.std(byte_accuracies) for b in byte_accuracies],
                              alpha=0.2, color='green')
        axes1[1].set_xlabel('Run Number')
        axes1[1].set_ylabel('Byte Accuracy')
        axes1[1].set_title('Byte Accuracy Across Runs')
        axes1[1].grid(True, alpha=0.3)
        axes1[1].legend()
        axes1[1].set_ylim(0, 1)

        # Success rate plot
        axes1[2].bar(run_numbers, success_rates, color='orange', alpha=0.7)
        axes1[2].axhline(y=np.mean(success_rates), color='r', linestyle='--',
                         label=f'Mean: {np.mean(success_rates):.2%}')
        axes1[2].set_xlabel('Run Number')
        axes1[2].set_ylabel('Success Rate (>70% pair acc)')
        axes1[2].set_title('Success Rate Across Runs')
        axes1[2].grid(True, alpha=0.3, axis='y')
        axes1[2].legend()
        axes1[2].set_ylim(0, 1)

        fig1.tight_layout()
        plot1_path = output_dir / f"accuracy_trends_{timestamp}.png"
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plot_files.append(plot1_path)

        # 2. Box plot of all runs
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Prepare data for box plot
        all_pair_accuracies = []
        for run in all_runs:
            run_accuracies = [r['pair_accuracy'] for r in run['individual_results']]
            all_pair_accuracies.append(run_accuracies)

        bp = ax2.boxplot(all_pair_accuracies, labels=run_numbers, patch_artist=True)

        # Customize box plot
        for box in bp['boxes']:
            box.set(facecolor='lightblue', alpha=0.7)
        for median in bp['medians']:
            median.set(color='red', linewidth=2)

        ax2.set_xlabel('Run Number')
        ax2.set_ylabel('Pair Accuracy')
        ax2.set_title('Distribution of Pair Accuracies in Each Run')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)

        fig2.tight_layout()
        plot2_path = output_dir / f"accuracy_distribution_{timestamp}.png"
        fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plot_files.append(plot2_path)

        # 3. Summary statistics bar chart
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        metrics_to_plot = ['avg_pair_accuracy', 'avg_byte_accuracy', 'success_rate']
        metric_names = ['Pair Accuracy', 'Byte Accuracy', 'Success Rate']
        colors = ['skyblue', 'lightgreen', 'salmon']

        x_pos = np.arange(len(all_runs))
        bar_width = 0.25

        for i, (metric, name, color) in enumerate(zip(metrics_to_plot, metric_names, colors)):
            values = [r['avg_metrics'][metric] for r in all_runs]
            ax3.bar(x_pos + i * bar_width, values, bar_width,
                    label=name, color=color, alpha=0.7)

        ax3.set_xlabel('Run Number')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Across Runs')
        ax3.set_xticks(x_pos + bar_width)
        ax3.set_xticklabels(run_numbers)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1)

        # Add value labels on top of bars
        for i, run in enumerate(all_runs):
            for j, metric in enumerate(metrics_to_plot):
                value = run['avg_metrics'][metric]
                ax3.text(i + j * bar_width, value + 0.01, f'{value:.2%}',
                         ha='center', va='bottom', fontsize=8)

        fig3.tight_layout()
        plot3_path = output_dir / f"performance_summary_{timestamp}.png"
        fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plot_files.append(plot3_path)

        plt.close('all')
        logger.info(f"Created {len(plot_files)} visualization plots")
        return plot_files

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save experiment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON (human readable)
        json_path = output_dir / f"analysis_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert to serializable format
            json_data = json.dumps(results, default=str, indent=2)
            f.write(json_data)

        # Save as pickle (preserves full structure)
        pickle_path = output_dir / f"analysis_results_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        # Create a summary text file
        summary_path = output_dir / f"analysis_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("TWO-TIME PAD ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")

            config = results.get('config', {})
            f.write(f"CONFIGURATION:\n")
            f.write(f"  Model n-gram size: {config.get('n', 'N/A')}\n")
            f.write(f"  Beam width: {config.get('beam_width', 'N/A')}\n")
            f.write(f"  Number of runs: {config.get('num_runs', 'N/A')}\n")
            f.write(f"  Tests per run: {config.get('num_tests_per_run', 'N/A')}\n")
            f.write(f"  Total emails used: {config.get('total_emails', 'N/A')}\n\n")

            overall = results.get('overall_metrics', {})
            f.write(f"OVERALL RESULTS:\n")
            f.write(f"  Average Pair Accuracy: {overall.get('overall_avg_avg_pair_accuracy', 0):.2%}\n")
            f.write(f"  Std Dev Pair Accuracy: {overall.get('overall_std_avg_pair_accuracy', 0):.4f}\n")
            f.write(f"  Average Byte Accuracy: {overall.get('overall_avg_avg_byte_accuracy', 0):.2%}\n")
            f.write(f"  Average Success Rate: {overall.get('overall_avg_success_rate', 0):.2%}\n\n")

            f.write("DETAILED RUN RESULTS:\n")
            for i, run in enumerate(results.get('all_runs', []), 1):
                metrics = run.get('avg_metrics', {})
                f.write(f"  Run {i}:\n")
                f.write(f"    Pair Accuracy: {metrics.get('avg_pair_accuracy', 0):.2%}\n")
                f.write(f"    Byte Accuracy: {metrics.get('avg_byte_accuracy', 0):.2%}\n")
                f.write(f"    Success Rate: {metrics.get('success_rate', 0):.2%}\n")
                f.write(f"    Tests run: {metrics.get('num_tests', 0)}\n\n")

        logger.info(f"Results saved to {output_dir}")
        return [json_path, pickle_path, summary_path]

    def analyze(self, num_emails: int = 10000, n: int = 7, beam_width: int = 100,
                num_runs: int = 5, num_tests_per_run: int = 10):
        """Main analysis pipeline"""
        logger.info("Starting analysis pipeline...")

        # Step 1: Ensure data is prepared
        logger.info("Step 1: Checking data...")
        data_dir = self.ensure_data_prepared(num_emails=num_emails)

        # Step 2: Load emails
        logger.info("Step 2: Loading emails...")
        emails = self.load_emails(data_dir, max_emails=num_emails)

        if len(emails) < 100:
            raise ValueError(f"Not enough emails loaded: {len(emails)}. Need at least 100.")

        # Step 3: Run cross-validation experiments
        logger.info("Step 3: Running experiments...")
        results = self.run_cross_validation(
            emails=emails,
            n=n,
            beam_width=beam_width,
            num_runs=num_runs,
            num_tests_per_run=num_tests_per_run
        )

        # Step 4: Create visualizations
        logger.info("Step 4: Creating visualizations...")
        plot_files = self.create_visualizations(results, self.results_dir)

        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        saved_files = self.save_results(results, self.results_dir)

        # Step 6: Print summary
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)

        overall = results['overall_metrics']
        logger.info(f"Overall Pair Accuracy: {overall['overall_avg_avg_pair_accuracy']:.2%} "
                    f"(±{overall['overall_std_avg_pair_accuracy']:.4f})")
        logger.info(f"Overall Byte Accuracy: {overall['overall_avg_avg_byte_accuracy']:.2%}")
        logger.info(f"Overall Success Rate: {overall['overall_avg_success_rate']:.2%}")

        logger.info(f"\nFiles created:")
        for file in plot_files + saved_files:
            logger.info(f"  - {file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Two-Time Pad Decryption Model Performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with default parameters
  py analysis

  # Analyze with 5000 emails, 10 runs
  py analysis --num-emails 5000 --num-runs 10

  # Use different model parameters
  py analysis --n 6 --beam-width 200 --num-tests 15

  # Quick test with minimal parameters
  py analysis --num-emails 1000 --num-runs 3 --num-tests 5
        """
    )

    parser.add_argument('--num-emails', type=int, default=10000,
                        help='Number of emails to use (default: 10000)')
    parser.add_argument('--n', type=int, default=7,
                        help='N-gram size for language model (default: 7)')
    parser.add_argument('--beam-width', type=int, default=100,
                        help='Beam width for decoder (default: 100)')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of cross-validation runs (default: 5)')
    parser.add_argument('--num-tests', type=int, default=10,
                        help='Number of tests per run (default: 10)')
    parser.add_argument('--res-dir', default='res',
                        help='Resources directory (default: res)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        analyzer = EmailAnalyzer(Path(args.res_dir))
        results = analyzer.analyze(
            num_emails=args.num_emails,
            n=args.n,
            beam_width=args.beam_width,
            num_runs=args.num_runs,
            num_tests_per_run=args.num_tests
        )

        # Exit with success
        sys.exit(0)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()