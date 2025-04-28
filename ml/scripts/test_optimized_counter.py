"""
Test the optimized CMU syllable counter interactively.
"""
import time
from rich.console import Console
from rich.table import Table

# Import the optimized CMU syllable counter
import sys
import os
# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimized_cmu_counter import (
    download_cmudict,
    load_cmu_dict,
    count_syllables_cmu,
    count_syllables_rule_based,
    count_syllables_hybrid
)

def interactive_test():
    """
    Interactive testing of syllable counters.
    """
    console = Console()
    
    console.print("[bold blue]Optimized CMU Syllable Counter[/bold blue]")
    console.print("[bold blue]=============================[/bold blue]")
    console.print("Enter words to count syllables (or 'q' to quit)")
    console.print("Special commands:")
    console.print("  [cyan]bench[/cyan] - Run benchmark on common words")
    
    # Download CMU dict
    download_cmudict()
    
    # Load CMU dict
    load_cmu_dict()
    
    while True:
        word = console.input("\n[bold green]> [/bold green]").strip()
        
        if word.lower() == 'q':
            break
        elif word.lower() == 'bench':
            run_benchmark(console)
            continue
        
        start_time = time.time()
        cmu_count = count_syllables_cmu(word)
        cmu_time = time.time() - start_time
        
        start_time = time.time()
        rule_count = count_syllables_rule_based(word)
        rule_time = time.time() - start_time
        
        start_time = time.time()
        hybrid_count = count_syllables_hybrid(word)
        hybrid_time = time.time() - start_time
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Method", style="dim")
        table.add_column("Syllables", justify="right")
        table.add_column("Time", justify="right")
        
        if cmu_count is not None:
            table.add_row("CMU Dict", str(cmu_count), f"{cmu_time*1000:.2f}ms")
        else:
            table.add_row("CMU Dict", "Not found", f"{cmu_time*1000:.2f}ms")
            
        table.add_row("Rule-based", str(rule_count), f"{rule_time*1000:.2f}ms")
        table.add_row("Hybrid", str(hybrid_count), f"{hybrid_time*1000:.2f}ms")
        
        console.print(table)

def run_benchmark(console):
    """
    Run benchmark on common words.
    """
    console.print("[bold blue]Running benchmark...[/bold blue]")
    
    # List of common words to benchmark
    words = [
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
        "his", "they", "her", "she", "will", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
        "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
        "elephant", "giraffe", "hippopotamus", "kangaroo", "leopard",
        "monkey", "octopus", "penguin", "rhinoceros", "tiger",
        "university", "laboratory", "dictionary", "encyclopedia", "mathematics",
        "philosophy", "psychology", "sociology", "anthropology", "archaeology",
        "beautiful", "colorful", "delightful", "elegant", "fascinating",
        "gorgeous", "harmonious", "incredible", "magnificent", "wonderful",
        "absolutely", "beautifully", "carefully", "delicately", "efficiently",
        "gracefully", "helpfully", "intelligently", "joyfully", "kindly",
        "communication", "determination", "entertainment", "fascination", "generation",
        "imagination", "motivation", "observation", "preparation", "satisfaction",
        "understanding", "appreciation", "celebration", "demonstration", "examination",
        "investigation", "organization", "participation", "recommendation", "transformation",
        "winebarger", "nickelodeon", "collaborationist", "authoritarianism",
        "supercalifragilistic", "internationalization", "institutionalization",
        "counterrevolutionary", "deinstitutionalization", "extraterritoriality",
        "antidisestablishmentarianism"
    ]
    
    # Run benchmark
    start_time = time.time()
    cmu_counts = []
    for word in words:
        cmu_counts.append(count_syllables_cmu(word))
    cmu_time = time.time() - start_time
    
    start_time = time.time()
    rule_counts = []
    for word in words:
        rule_counts.append(count_syllables_rule_based(word))
    rule_time = time.time() - start_time
    
    start_time = time.time()
    hybrid_counts = []
    for word in words:
        hybrid_counts.append(count_syllables_hybrid(word))
    hybrid_time = time.time() - start_time
    
    # Run second pass to test caching
    start_time = time.time()
    for word in words:
        count_syllables_hybrid(word)
    cached_time = time.time() - start_time
    
    # Print results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim")
    table.add_column("Total Time", justify="right")
    table.add_column("Words/sec", justify="right")
    
    table.add_row("CMU Dict", f"{cmu_time:.4f}s", f"{len(words)/cmu_time:.2f}")
    table.add_row("Rule-based", f"{rule_time:.4f}s", f"{len(words)/rule_time:.2f}")
    table.add_row("Hybrid", f"{hybrid_time:.4f}s", f"{len(words)/hybrid_time:.2f}")
    table.add_row("Cached (2nd pass)", f"{cached_time:.4f}s", f"{len(words)/cached_time:.2f}")
    
    console.print(table)
    
    # Print some example results
    console.print("\n[bold blue]Example results:[/bold blue]")
    example_table = Table(show_header=True, header_style="bold magenta")
    example_table.add_column("Word", style="dim")
    example_table.add_column("CMU", justify="right")
    example_table.add_column("Rule", justify="right")
    example_table.add_column("Hybrid", justify="right")
    
    for i in range(min(10, len(words))):
        example_table.add_row(
            words[i],
            str(cmu_counts[i]) if cmu_counts[i] is not None else "Not found",
            str(rule_counts[i]),
            str(hybrid_counts[i])
        )
    
    console.print(example_table)

if __name__ == "__main__":
    interactive_test()
