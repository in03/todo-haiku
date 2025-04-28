"""
Interactive testing script for the character-level LSTM syllable counter model.
"""
import torch
import json
import time
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
import statistics

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

# Import the CharLSTM model
from train_char_lstm import CharLSTM, count_syllables_from_bio


@dataclass
class PredictionMetrics:
    word: str
    syllables: int
    cmu_syllables: int
    rule_based_syllables: int
    bio_tags: list
    total_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float


class SyllablePredictor:
    def __init__(self, model_path="models/syllable_char_lstm.pt", vocab_path="models/char_vocab_lstm.json"):
        self.console = Console(width=80)  # Set fixed width
        self.metrics_history: list[PredictionMetrics] = []
        self.model, self.vocab = self.load_model_and_vocab(model_path, vocab_path)
        
    def load_model_and_vocab(self, model_path, vocab_path):
        start = time.perf_counter()
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = CharLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=3,  # B, I, O
            n_layers=checkpoint['n_layers'],
            dropout=checkpoint['dropout']
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
        chars = [self.vocab.get(c.lower(), self.vocab.get('<unk>', 1)) for c in word]
        char_tensor = torch.tensor(chars).unsqueeze(0)
        length_tensor = torch.tensor([len(word)])
        prep_time = (time.perf_counter() - start_prep) * 1000
        
        # Inference
        start_inference = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(char_tensor, length_tensor)
            
            # Get the predicted BIO tags
            _, predicted = torch.max(outputs.squeeze(0), 1)
            bio_tags = predicted.tolist()
            
            # Count syllables from BIO tags
            syllable_count = count_syllables_from_bio(bio_tags)
            
            print(f"\nPredicted BIO tags for '{word}':")
            print(f"Word:  {' '.join(word)}")
            print(f"Tags:  {' '.join(str(tag) for tag in bio_tags)}")
            print(f"Count: {syllable_count} syllables")
            
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
            syllables=syllable_count,
            cmu_syllables=cmu_syl,
            rule_based_syllables=rule_based_syl,
            bio_tags=bio_tags,
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
        self.console.print("  [cyan]test[/cyan] - Run test on common words")
        
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
                elif word.lower() == 'test':
                    self.run_test()
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
    
    def run_test(self):
        """
        Run a test on common words.
        """
        self.console.print("[bold blue]Running test on common words...[/bold blue]")
        
        # List of test words
        test_words = [
            # Common one-syllable words
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
            
            # Common two-syllable words
            "about", "people", "into", "other", "only", "after", "because", "even",
            
            # Common three-syllable words
            "beautiful", "important", "together", "wonderful", "tomorrow", "yesterday",
            
            # Common four-syllable words
            "opportunity", "individual", "television", "information", "understanding",
            
            # Common five-syllable words
            "university", "responsibility", "international", "communication", "imagination",
            
            # Problematic words
            "shoe", "potato", "elephant", "winebarger", "nickelodeon", "collaborationist",
            "authoritarianism", "supercalifragilistic", "internationalization",
            "institutionalization", "counterrevolutionary", "deinstitutionalization",
            "extraterritoriality", "antidisestablishmentarianism"
        ]
        
        # Run test
        results = []
        for word in test_words:
            metrics = self.predict_word(word)
            results.append(metrics)
        
        # Calculate statistics
        cmu_words = [m for m in results if m.cmu_syllables != -1]
        if cmu_words:
            correct_cmu = sum(1 for m in cmu_words if m.syllables == m.cmu_syllables)
            accuracy_cmu = correct_cmu / len(cmu_words)
            avg_abs_diff_cmu = sum(abs(m.syllables - m.cmu_syllables) for m in cmu_words) / len(cmu_words)
        else:
            accuracy_cmu = 0
            avg_abs_diff_cmu = 0
        
        correct_rule = sum(1 for m in results if m.syllables == m.rule_based_syllables)
        accuracy_rule = correct_rule / len(results)
        avg_abs_diff_rule = sum(abs(m.syllables - m.rule_based_syllables) for m in results) / len(results)
        
        # Print results
        table = Table(show_header=True, header_style="bold blue", width=80)
        table.add_column("Word", style="dim", width=25)
        table.add_column("Model", justify="right", width=10)
        table.add_column("CMU", justify="right", width=10)
        table.add_column("Rule", justify="right", width=10)
        table.add_column("Diff (CMU)", justify="right", width=10)
        
        for metrics in results:
            if metrics.cmu_syllables != -1:
                diff_cmu = metrics.syllables - metrics.cmu_syllables
                if diff_cmu == 0:
                    table.add_row(
                        metrics.word,
                        f"[green]{metrics.syllables}[/green]",
                        str(metrics.cmu_syllables),
                        str(metrics.rule_based_syllables),
                        f"[green]{diff_cmu}[/green]"
                    )
                elif abs(diff_cmu) == 1:
                    table.add_row(
                        metrics.word,
                        f"[yellow]{metrics.syllables}[/yellow]",
                        str(metrics.cmu_syllables),
                        str(metrics.rule_based_syllables),
                        f"[yellow]{diff_cmu}[/yellow]"
                    )
                else:
                    table.add_row(
                        metrics.word,
                        f"[red]{metrics.syllables}[/red]",
                        str(metrics.cmu_syllables),
                        str(metrics.rule_based_syllables),
                        f"[red]{diff_cmu}[/red]"
                    )
            else:
                diff_rule = metrics.syllables - metrics.rule_based_syllables
                if diff_rule == 0:
                    table.add_row(
                        metrics.word,
                        f"[green]{metrics.syllables}[/green]",
                        "N/A",
                        str(metrics.rule_based_syllables),
                        "N/A"
                    )
                elif abs(diff_rule) == 1:
                    table.add_row(
                        metrics.word,
                        f"[yellow]{metrics.syllables}[/yellow]",
                        "N/A",
                        str(metrics.rule_based_syllables),
                        "N/A"
                    )
                else:
                    table.add_row(
                        metrics.word,
                        f"[red]{metrics.syllables}[/red]",
                        "N/A",
                        str(metrics.rule_based_syllables),
                        "N/A"
                    )
        
        self.console.print(table)
        
        # Print statistics
        stats_table = Table(show_header=True, header_style="bold blue", width=50)
        stats_table.add_column("Statistic", style="dim", width=30)
        stats_table.add_column("Value", justify="right", width=15)
        
        stats_table.add_row("Words Tested", str(len(results)))
        stats_table.add_row("Accuracy (CMU)", f"{accuracy_cmu:.2%}")
        stats_table.add_row("Avg Abs Diff (CMU)", f"{avg_abs_diff_cmu:.2f}")
        stats_table.add_row("Accuracy (Rule)", f"{accuracy_rule:.2%}")
        stats_table.add_row("Avg Abs Diff (Rule)", f"{avg_abs_diff_rule:.2f}")
        
        self.console.print(stats_table)


def main():
    # Download CMU dict if needed
    download_cmudict()
    
    predictor = SyllablePredictor()
    predictor.run_interactive()


if __name__ == "__main__":
    main()
