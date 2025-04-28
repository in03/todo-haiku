"""
Test the CMU syllable counter interactively.
"""
import nltk
from rich.console import Console
from rich.table import Table

# Download the CMU Pronouncing Dictionary if not already downloaded
def download_cmudict():
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        print("Downloading CMU Pronouncing Dictionary...")
        nltk.download('cmudict')

# Count syllables using the CMU Pronouncing Dictionary
def count_syllables_cmu(word):
    """
    Count syllables using the CMU Pronouncing Dictionary.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables or None if word not in dictionary
    """
    import re
    word = word.lower().strip()
    
    # Remove punctuation
    word = re.sub(r'[^\w\s]', '', word)
    
    # Get pronunciations from CMU dict
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()
        
        if word in d:
            # Count vowel phonemes (those containing a digit)
            return max([len([ph for ph in phones if re.search(r'\d', ph)]) for phones in d[word]])
        else:
            return None
    except Exception as e:
        print(f"Error using CMU dict: {e}")
        return None

# Rule-based syllable counter as fallback
def count_syllables_rule_based(word):
    """
    Count syllables using a rule-based approach.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables
    """
    import re
    # Remove punctuation and convert to lowercase
    word = re.sub(r'[.,;:!?()\'"]', '', word.lower())
    
    # Special cases dictionary
    special_cases = {
        'eye': 1, 'eyes': 1, 'queue': 1, 'queues': 1, 'quay': 1, 'quays': 1,
        'business': 2, 'businesses': 3, 'colonel': 2, 'colonels': 2,
        'island': 2, 'islands': 2, 'recipe': 3, 'recipes': 3,
        'wednesday': 3, 'wednesdays': 3, 'area': 3, 'areas': 3,
        'idea': 3, 'ideas': 3,
    }
    
    # Check for special cases
    if word in special_cases:
        return special_cases[word]
    
    # Count syllables based on vowel groups
    vowels = 'aeiouy'
    count = 0
    prev_is_vowel = False
    
    # Handle specific patterns
    if word and vowels.find(word[0]) >= 0:
        count = 1
        prev_is_vowel = True
    
    for i in range(1, len(word)):
        is_vowel = vowels.find(word[i]) >= 0
        
        if is_vowel and not prev_is_vowel:
            count += 1
        
        prev_is_vowel = is_vowel
    
    # Handle silent e at the end
    if len(word) > 2 and word.endswith('e') and not word[-2] in vowels:
        count = max(1, count - 1)
    
    # Handle words ending with 'le' where the 'l' is preceded by a consonant
    if len(word) > 2 and word.endswith('le') and not word[-3] in vowels:
        count += 1
    
    # Handle words ending with 'ed'
    if len(word) > 2 and word.endswith('ed'):
        # Only count as a syllable if preceded by t or d
        if word[-3] not in ['t', 'd']:
            count = max(1, count - 1)
    
    # Ensure at least one syllable
    return max(1, count)

# Hybrid syllable counter that uses CMU dict when available, falls back to rule-based
def count_syllables_hybrid(word):
    """
    Count syllables using CMU dict when available, falling back to rule-based.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables
    """
    cmu_count = count_syllables_cmu(word)
    if cmu_count is not None:
        return cmu_count
    else:
        return count_syllables_rule_based(word)

# Interactive testing
def interactive_test():
    """
    Interactive testing of syllable counters.
    """
    console = Console()
    
    console.print("[bold blue]CMU Syllable Counter[/bold blue]")
    console.print("[bold blue]===================[/bold blue]")
    console.print("Enter words to count syllables (or 'q' to quit)")
    
    while True:
        word = console.input("\n[bold green]> [/bold green]").strip()
        
        if word.lower() == 'q':
            break
        
        cmu_count = count_syllables_cmu(word)
        rule_count = count_syllables_rule_based(word)
        hybrid_count = count_syllables_hybrid(word)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Method", style="dim")
        table.add_column("Syllables", justify="right")
        
        if cmu_count is not None:
            table.add_row("CMU Dict", str(cmu_count))
        else:
            table.add_row("CMU Dict", "Not found")
            
        table.add_row("Rule-based", str(rule_count))
        table.add_row("Hybrid", str(hybrid_count))
        
        console.print(table)

if __name__ == "__main__":
    # Download CMU dict
    download_cmudict()
    
    # Run interactive test
    interactive_test()
