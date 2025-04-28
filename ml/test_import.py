"""
Test importing the syllable_counter package.
"""
try:
    import syllable_counter
    print(f"Successfully imported syllable_counter version: {syllable_counter.__version__}")
    print("Package structure:")
    print(f"- preprocessing: {dir(syllable_counter.preprocessing)}")
    print(f"- training: {dir(syllable_counter.training)}")
    print(f"- evaluation: {dir(syllable_counter.evaluation)}")
except ImportError as e:
    print(f"Error importing syllable_counter: {e}")
