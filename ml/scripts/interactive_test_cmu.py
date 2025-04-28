"""
Interactive testing script for the CMU-based syllable counter model.
"""
import torch
import json
import time
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
import statistics

# Import the model and preprocessing functions
from syllable_counter.training import SyllableCounter
from syllable_counter.preprocessing import prepare_features

# Import the CMU syllable counter for comparison
import sys
import os
# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try to import from optimized version first, fall back to original if not available
try:
    from optimized_cmu_counter import count_syllables_cmu, count_syllables_rule_based, count_syllables_hybrid, download_cmudict
    print("Using optimized CMU syllable counter")
except ImportError:
    from cmu_syllable_counter import count_syllables_cmu, count_syllables_rule_based, count_syllables_hybrid, download_cmudict
    print("Using original CMU syllable counter")


@dataclass
class PredictionMetrics:
    word: str
    syllables: int
    cmu_syllables: int
    rule_based_syllables: int
    total_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float


class SyllablePredictor:
    def __init__(self, model_path="models/syllable_counter_cmu.pt", vocab_path="models/char_vocab_cmu.json"):
        self.console = Console(width=80)  # Set fixed width
        self.metrics_history: list[PredictionMetrics] = []
        self.model, self.vocab = self.load_model_and_vocab(model_path, vocab_path)

    def load_model_and_vocab(self, model_path, vocab_path):
        start = time.perf_counter()

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SyllableCounter(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_features=checkpoint['num_features']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        load_time = (time.perf_counter() - start) * 1000
        self.console.print(f"[green]Model loaded in {load_time:.2f}ms[/green]")

        return model, vocab

    def predict_word(self, word: str) -> PredictionMetrics:
        start_total = time.perf_counter()

        # Preprocessing
        start_prep = time.perf_counter()
        chars = [self.vocab.get(c.lower(), 0) for c in word]
        char_tensor = torch.tensor(chars).unsqueeze(0)
        features_df = prepare_features([word])
        feature_tensor = torch.tensor(features_df.drop('word', axis=1).values, dtype=torch.float32)
        prep_time = (time.perf_counter() - start_prep) * 1000

        # Inference
        start_inference = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(char_tensor, feature_tensor)
            print(f"\nRaw logits for '{word}':")

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Print logits and probabilities side by side
            print("\nSyllable | Logit  | Probability")
            print("-" * 35)
            for i, (logit, prob) in enumerate(zip(outputs[0], probs[0])):
                print(f"{i:8d} | {logit:6.2f} | {prob:10.4f}")

            # Get the predicted syllable count
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()

            print(f"\nModel prediction: {predicted} syllables")

        inference_time = (time.perf_counter() - start_inference) * 1000

        # Postprocessing
        start_post = time.perf_counter()
        # Get CMU and rule-based syllable counts for comparison
        cmu_syl = count_syllables_cmu(word)
        if cmu_syl is None:
            cmu_syl = -1  # Indicate not found in CMU dict
        rule_based_syl = count_syllables_rule_based(word)
        post_time = (time.perf_counter() - start_post) * 1000

        total_time = (time.perf_counter() - start_total) * 1000

        metrics = PredictionMetrics(
            word=word,
            syllables=predicted,
            cmu_syllables=cmu_syl,
            rule_based_syllables=rule_based_syl,
            total_time_ms=total_time,
            preprocessing_time_ms=prep_time,
            inference_time_ms=inference_time,
            postprocessing_time_ms=post_time
        )

        self.metrics_history.append(metrics)
        return metrics

    def display_metrics(self, metrics: PredictionMetrics):
        table = Table(show_header=True, header_style="bold magenta", width=50)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right", width=15)

        table.add_row("Word", metrics.word)

        # Highlight if the model prediction matches the CMU count
        if metrics.cmu_syllables != -1:
            if metrics.syllables == metrics.cmu_syllables:
                table.add_row("Syllables (Model)", f"[green]{metrics.syllables}[/green]")
            else:
                table.add_row("Syllables (Model)", f"[yellow]{metrics.syllables}[/yellow]")
            table.add_row("Syllables (CMU)", str(metrics.cmu_syllables))

            # Show difference with CMU
            diff_cmu = metrics.syllables - metrics.cmu_syllables
            if diff_cmu == 0:
                table.add_row("Diff (Model-CMU)", f"[green]{diff_cmu}[/green]")
            elif abs(diff_cmu) == 1:
                table.add_row("Diff (Model-CMU)", f"[yellow]{diff_cmu}[/yellow]")
            else:
                table.add_row("Diff (Model-CMU)", f"[red]{diff_cmu}[/red]")
        else:
            table.add_row("Syllables (Model)", str(metrics.syllables))
            table.add_row("Syllables (CMU)", "Not in CMU dict")

        # Show rule-based count
        if metrics.syllables == metrics.rule_based_syllables:
            table.add_row("Syllables (Rule)", f"[green]{metrics.rule_based_syllables}[/green]")
        else:
            table.add_row("Syllables (Rule)", str(metrics.rule_based_syllables))

        # Show difference with rule-based
        diff_rule = metrics.syllables - metrics.rule_based_syllables
        if diff_rule == 0:
            table.add_row("Diff (Model-Rule)", f"[green]{diff_rule}[/green]")
        elif abs(diff_rule) == 1:
            table.add_row("Diff (Model-Rule)", f"[yellow]{diff_rule}[/yellow]")
        else:
            table.add_row("Diff (Model-Rule)", f"[red]{diff_rule}[/red]")

        table.add_row("Total Time", f"{metrics.total_time_ms:.1f}ms")
        table.add_row("Pre-process", f"{metrics.preprocessing_time_ms:.1f}ms")
        table.add_row("Inference", f"{metrics.inference_time_ms:.1f}ms")
        table.add_row("Post-process", f"{metrics.postprocessing_time_ms:.1f}ms")

        if len(self.metrics_history) > 1:
            table.add_row("", "")  # Empty row as separator
            table.add_row(
                "Avg Total",
                f"{statistics.mean(m.total_time_ms for m in self.metrics_history):.1f}ms"
            )
            table.add_row(
                "Avg Inference",
                f"{statistics.mean(m.inference_time_ms for m in self.metrics_history):.1f}ms"
            )

        self.console.print(table)

    def run_interactive(self):
        self.console.print("[bold blue]Enter words to count syllables (or 'q' to quit)[/bold blue]")
        self.console.print("[bold blue]Special commands:[/bold blue]")
        self.console.print("  [cyan]stats[/cyan] - Show detailed statistics")
        self.console.print("  [cyan]clear[/cyan] - Clear metrics history")

        while True:
            try:
                word = self.console.input("\n[bold green]> [/bold green]").strip()

                if word.lower() == 'q':
                    break
                elif word.lower() == 'stats':
                    self.show_statistics()
                    continue
                elif word.lower() == 'clear':
                    self.metrics_history.clear()
                    self.console.print("[yellow]Metrics history cleared[/yellow]")
                    continue

                metrics = self.predict_word(word)
                self.display_metrics(metrics)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

    def show_statistics(self):
        if not self.metrics_history:
            self.console.print("[yellow]No predictions made yet[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue", width=50)
        table.add_column("Statistic", style="dim", width=20)
        table.add_column("Value", justify="right", width=15)

        times = [(m.word, m.syllables, m.cmu_syllables, m.rule_based_syllables, m.total_time_ms) for m in self.metrics_history]

        table.add_row("Words Tested", str(len(times)))

        # Calculate accuracy against CMU dict
        cmu_words = [m for m in self.metrics_history if m.cmu_syllables != -1]
        if cmu_words:
            correct_cmu = sum(1 for m in cmu_words if m.syllables == m.cmu_syllables)
            accuracy_cmu = correct_cmu / len(cmu_words)
            table.add_row("Accuracy (CMU)", f"{accuracy_cmu:.2%}")

            # Calculate average absolute difference
            avg_abs_diff_cmu = sum(abs(m.syllables - m.cmu_syllables) for m in cmu_words) / len(cmu_words)
            table.add_row("Avg Abs Diff (CMU)", f"{avg_abs_diff_cmu:.2f}")

        # Calculate accuracy against rule-based
        correct_rule = sum(1 for m in self.metrics_history if m.syllables == m.rule_based_syllables)
        accuracy_rule = correct_rule / len(self.metrics_history)
        table.add_row("Accuracy (Rule)", f"{accuracy_rule:.2%}")

        # Calculate average absolute difference
        avg_abs_diff_rule = sum(abs(m.syllables - m.rule_based_syllables) for m in self.metrics_history) / len(self.metrics_history)
        table.add_row("Avg Abs Diff (Rule)", f"{avg_abs_diff_rule:.2f}")

        table.add_row("", "")  # Separator

        # Show last 5 predictions
        table.add_row("Recent Predictions:", "")
        for word, model_syls, cmu_syls, rule_syls, ms in times[-5:]:
            if cmu_syls != -1:
                diff_cmu = model_syls - cmu_syls
                if diff_cmu == 0:
                    table.add_row(f"  {word}", f"[green]{model_syls}[/green] ({ms:.1f}ms)")
                else:
                    table.add_row(f"  {word}", f"{model_syls} vs CMU:{cmu_syls} ({diff_cmu:+d}) ({ms:.1f}ms)")
            else:
                diff_rule = model_syls - rule_syls
                if diff_rule == 0:
                    table.add_row(f"  {word}", f"[green]{model_syls}[/green] ({ms:.1f}ms)")
                else:
                    table.add_row(f"  {word}", f"{model_syls} vs Rule:{rule_syls} ({diff_rule:+d}) ({ms:.1f}ms)")

        self.console.print(table)


def main():
    # Download CMU dict if needed
    download_cmudict()

    predictor = SyllablePredictor()
    predictor.run_interactive()


if __name__ == "__main__":
    main()
