import torch
from pathlib import Path
import json
import time
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
import statistics

# Import the model and preprocessing functions
from syllable_counter.training import SyllableCounter
from syllable_counter.preprocessing import prepare_features

@dataclass
class PredictionMetrics:
    word: str
    syllables: int
    total_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float

class SyllablePredictor:
    def __init__(self):
        self.console = Console(width=80)  # Set fixed width
        self.metrics_history: list[PredictionMetrics] = []
        self.model, self.vocab = self.load_model_and_vocab()
        
    def load_model_and_vocab(self):
        start = time.perf_counter()
        
        # Load model
        checkpoint = torch.load('models/syllable_counter.pt', map_location='cpu')
        model = SyllableCounter(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_features=checkpoint['num_features']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load vocabulary
        with open('models/char_vocab.json', 'r') as f:
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
            
            # Get both raw and bias-corrected predictions
            _, raw_pred = torch.max(outputs, dim=1)
            raw_pred = raw_pred.item()
            
            # Apply bias correction (subtract 2 from prediction)
            corrected_pred = max(1, raw_pred - 2)
            
            print(f"\nRaw prediction: {raw_pred} syllables")
            print(f"Corrected prediction: {corrected_pred} syllables")
            
            predicted = corrected_pred
            
        inference_time = (time.perf_counter() - start_inference) * 1000
        
        # Postprocessing
        start_post = time.perf_counter()
        post_time = (time.perf_counter() - start_post) * 1000
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        metrics = PredictionMetrics(
            word=word,
            syllables=predicted,
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
        table.add_row("Syllables", str(metrics.syllables))
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
        
        times = [(m.word, m.syllables, m.total_time_ms) for m in self.metrics_history]
        
        table.add_row("Words Tested", str(len(times)))
        table.add_row("", "")  # Separator
        
        # Show last 5 predictions
        table.add_row("Recent Predictions:", "")
        for word, syls, ms in times[-5:]:
            table.add_row(f"  {word}", f"{syls} syls ({ms:.1f}ms)")
        
        self.console.print(table)

def main():
    predictor = SyllablePredictor()
    predictor.run_interactive()

if __name__ == "__main__":
    main()




