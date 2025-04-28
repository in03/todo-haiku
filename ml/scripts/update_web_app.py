"""
Update the web application to use the balanced syllable counter model.
"""
import os
import shutil
import json
from pathlib import Path


def update_web_app(
    model_dir="models",
    public_dir="../public/models",
    update_ts_files=True
):
    """
    Update the web application to use the balanced syllable counter model.
    
    Args:
        model_dir: Directory containing the model files
        public_dir: Public directory to copy the model files to
        update_ts_files: Whether to update the TypeScript files
    """
    # Create the public models directory if it doesn't exist
    os.makedirs(public_dir, exist_ok=True)
    
    # Copy the model files
    print(f"Copying model files from {model_dir} to {public_dir}...")
    
    # Copy the ONNX model
    shutil.copy(
        os.path.join(model_dir, "syllable_counter_balanced.onnx"),
        os.path.join(public_dir, "syllable_counter_balanced.onnx")
    )
    print(f"Copied syllable_counter_balanced.onnx")
    
    # Copy the vocabulary
    shutil.copy(
        os.path.join(model_dir, "char_vocab_balanced.json"),
        os.path.join(public_dir, "char_vocab_balanced.json")
    )
    print(f"Copied char_vocab_balanced.json")
    
    # Copy the metadata
    shutil.copy(
        os.path.join(model_dir, "model_metadata_balanced.json"),
        os.path.join(public_dir, "model_metadata_balanced.json")
    )
    print(f"Copied model_metadata_balanced.json")
    
    print("Model files copied successfully.")
    
    # Update the TypeScript files if requested
    if update_ts_files:
        update_typescript_files()


def update_typescript_files():
    """
    Update the TypeScript files to use the balanced syllable counter model.
    """
    # Define the files to update
    files_to_update = [
        "../src/utils/syllable-counter-switch.ts",
        "../src/utils/balanced-syllable-counter-onnx.ts"
    ]
    
    # Create the balanced syllable counter ONNX implementation
    create_balanced_counter_file()
    
    # Update the syllable counter switch
    update_counter_switch()
    
    print("TypeScript files updated successfully.")


def create_balanced_counter_file():
    """
    Create the balanced syllable counter ONNX implementation.
    """
    balanced_counter_path = "../src/utils/balanced-syllable-counter-onnx.ts"
    
    # Check if the file already exists
    if os.path.exists(balanced_counter_path):
        print(f"File {balanced_counter_path} already exists. Skipping creation.")
        return
    
    # Create the file
    with open(balanced_counter_path, "w") as f:
        f.write("""// Balanced ONNX-based syllable counter implementation
import { InferenceSession, Tensor } from 'onnxruntime-web';

// Define interface for model metadata
interface ModelMetadata {
  max_word_len: number;
  vocab_size: number;
  embedding_dim: number;
  hidden_dim: number;
  num_features: number;
  test_accuracy: number;
}

// Define interface for character vocabulary
interface CharVocabulary {
  [key: string]: number;
}

// Class for balanced ONNX-based syllable counting
export class BalancedOnnxSyllableCounter {
  private session: InferenceSession | null = null;
  private charToIdx: CharVocabulary | null = null;
  private metadata: ModelMetadata | null = null;
  private maxWordLen: number = 20;
  private isLoaded: boolean = false;
  private isLoading: boolean = false;
  private loadError: Error | null = null;

  // Initialize the model
  async init(): Promise<void> {
    if (this.isLoaded) {
      return;
    }

    if (this.isLoading) {
      // Wait for the current loading process to complete
      while (this.isLoading) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      if (this.loadError) {
        throw this.loadError;
      }
      
      return;
    }

    this.isLoading = true;
    this.loadError = null;

    try {
      // Load model metadata
      const metadataResponse = await fetch('/models/model_metadata_balanced.json');
      this.metadata = await metadataResponse.json();
      this.maxWordLen = this.metadata.max_word_len;

      // Load character vocabulary
      const vocabResponse = await fetch('/models/char_vocab_balanced.json');
      this.charToIdx = await vocabResponse.json();

      // Create ONNX session
      this.session = await InferenceSession.create('/models/syllable_counter_balanced.onnx');
      
      this.isLoaded = true;
      console.log('Balanced syllable counter model loaded successfully');
    } catch (error) {
      this.loadError = error as Error;
      console.error('Failed to load balanced syllable counter model:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // Count syllables in a word
  async countSyllablesInWord(word: string): Promise<number> {
    if (!this.isLoaded || !this.session || !this.charToIdx) {
      throw new Error('Model not loaded');
    }

    // Preprocess the word
    const wordLower = word.toLowerCase().replace(/[.,;:!?()'"]/g, '');

    // Extract features
    const features = this.extractFeatures(wordLower);

    // Convert word to character indices
    const charIds = new Array(this.maxWordLen).fill(0);
    for (let i = 0; i < Math.min(wordLower.length, this.maxWordLen); i++) {
      const char = wordLower[i];
      charIds[i] = this.charToIdx[char] || 1; // Use <unk> token (1) if character not in vocabulary
    }

    // Create input tensors
    const inputs = {
      'char_ids': new Tensor('int64', new BigInt64Array(charIds.map(n => BigInt(n))), [1, this.maxWordLen]),
      'word_length': new Tensor('float32', new Float32Array([features.wordLength]), [1, 1]),
      'num_vowels': new Tensor('float32', new Float32Array([features.numVowels]), [1, 1]),
      'num_consonants': new Tensor('float32', new Float32Array([features.numConsonants]), [1, 1]),
      'vowel_sequences': new Tensor('float32', new Float32Array([features.vowelSequences]), [1, 1]),
      'ends_with_e': new Tensor('float32', new Float32Array([features.endsWithE]), [1, 1]),
      'ends_with_le': new Tensor('float32', new Float32Array([features.endsWithLe]), [1, 1]),
      'ends_with_ed': new Tensor('float32', new Float32Array([features.endsWithEd]), [1, 1]),
    };

    // Run inference
    const outputMap = await this.session.run(inputs);
    const output = outputMap['syllable_count'];
    const predictions = output.data as Float32Array;

    // Get the predicted syllable count (argmax)
    let maxIndex = 0;
    let maxValue = predictions[0];
    for (let i = 1; i < predictions.length; i++) {
      if (predictions[i] > maxValue) {
        maxValue = predictions[i];
        maxIndex = i;
      }
    }

    // No bias correction needed for the balanced model
    return maxIndex;
  }

  // Extract features from a word
  private extractFeatures(word: string): {
    wordLength: number;
    numVowels: number;
    numConsonants: number;
    vowelSequences: number;
    endsWithE: number;
    endsWithLe: number;
    endsWithEd: number;
  } {
    const vowels = 'aeiouy';
    const consonants = 'bcdfghjklmnpqrstvwxz';

    // Basic features
    const wordLength = word.length;
    const numVowels = [...word].filter(c => vowels.includes(c)).length;
    const numConsonants = [...word].filter(c => consonants.includes(c)).length;

    // Vowel sequences
    let vowelSequences = 0;
    let inVowelSeq = false;
    for (const c of word) {
      if (vowels.includes(c)) {
        if (!inVowelSeq) {
          vowelSequences++;
          inVowelSeq = true;
        }
      } else {
        inVowelSeq = false;
      }
    }

    // Special patterns
    const endsWithE = word.endsWith('e') ? 1 : 0;
    const endsWithLe = word.endsWith('le') && word.length > 2 && !vowels.includes(word[word.length - 3]) ? 1 : 0;
    const endsWithEd = word.endsWith('ed') && word.length > 2 ? 1 : 0;

    return {
      wordLength,
      numVowels,
      numConsonants,
      vowelSequences,
      endsWithE,
      endsWithLe,
      endsWithEd
    };
  }

  // Count syllables in a line of text
  async countSyllables(text: string): Promise<number> {
    if (!text) return 0;

    // Split text into words
    const words = text.split(/\\s+/).filter(word => word.length > 0);

    // Count syllables in each word and sum them
    let totalSyllables = 0;
    for (const word of words) {
      totalSyllables += await this.countSyllablesInWord(word);
    }

    return totalSyllables;
  }
}

// Singleton instance
let instance: BalancedOnnxSyllableCounter | null = null;

// Get the singleton instance
export function getBalancedOnnxSyllableCounter(): BalancedOnnxSyllableCounter {
  if (!instance) {
    instance = new BalancedOnnxSyllableCounter();
  }
  return instance;
}
""")
    
    print(f"Created {balanced_counter_path}")


def update_counter_switch():
    """
    Update the syllable counter switch to include the balanced model.
    """
    counter_switch_path = "../src/utils/syllable-counter-switch.ts"
    
    # Check if the file exists
    if not os.path.exists(counter_switch_path):
        print(f"File {counter_switch_path} does not exist. Skipping update.")
        return
    
    # Read the file
    with open(counter_switch_path, "r") as f:
        content = f.read()
    
    # Check if the balanced model is already included
    if "BALANCED_ONNX" in content:
        print(f"Balanced model already included in {counter_switch_path}. Skipping update.")
        return
    
    # Update the file
    updated_content = content.replace(
        "export enum SyllableCounterType {",
        "export enum SyllableCounterType {\n  BALANCED_ONNX = 'BALANCED_ONNX',"
    )
    
    # Add import for balanced model
    updated_content = updated_content.replace(
        "import { getOnnxSyllableCounter } from './syllable-counter-onnx';",
        "import { getOnnxSyllableCounter } from './syllable-counter-onnx';\nimport { getBalancedOnnxSyllableCounter } from './balanced-syllable-counter-onnx';"
    )
    
    # Add initialization for balanced model
    if "let balancedOnnxInitialized = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let simpleOnnxInitialized = false;",
            "let simpleOnnxInitialized = false;\nlet balancedOnnxInitialized = false;"
        )
    
    if "let balancedOnnxInitializing = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let simpleOnnxInitializing = false;",
            "let simpleOnnxInitializing = false;\nlet balancedOnnxInitializing = false;"
        )
    
    if "let balancedOnnxInitError: Error | null = null;" not in updated_content:
        updated_content = updated_content.replace(
            "let simpleOnnxInitError: Error | null = null;",
            "let simpleOnnxInitError: Error | null = null;\nlet balancedOnnxInitError: Error | null = null;"
        )
    
    # Add initialization function for balanced model
    if "export async function initBalancedOnnxModel(): Promise<void> {" not in updated_content:
        init_function = """
// Initialize the Balanced ONNX model
export async function initBalancedOnnxModel(): Promise<void> {
  if (balancedOnnxInitialized || balancedOnnxInitializing) {
    return;
  }

  balancedOnnxInitializing = true;
  try {
    const balancedOnnxCounter = getBalancedOnnxSyllableCounter();
    await balancedOnnxCounter.init();
    balancedOnnxInitialized = true;
    balancedOnnxInitError = null;
  } catch (error) {
    balancedOnnxInitError = error as Error;
    console.error('Failed to initialize Balanced ONNX model:', error);
    // Fall back to rule-based counter
    currentCounterType = SyllableCounterType.RULE_BASED;
  } finally {
    balancedOnnxInitializing = false;
  }
}"""
        
        # Find the position to insert the function
        init_simple_pos = updated_content.find("export async function initSimpleOnnxModel()")
        if init_simple_pos != -1:
            # Find the end of the function
            init_simple_end = updated_content.find("}", init_simple_pos)
            if init_simple_end != -1:
                # Insert after the function
                updated_content = updated_content[:init_simple_end+1] + init_function + updated_content[init_simple_end+1:]
    
    # Update the countSyllables function
    if "else if (currentCounterType === SyllableCounterType.BALANCED_ONNX && balancedOnnxInitialized) {" not in updated_content:
        balanced_case = """
  else if (currentCounterType === SyllableCounterType.BALANCED_ONNX && balancedOnnxInitialized) {
    try {
      const balancedOnnxCounter = getBalancedOnnxSyllableCounter();
      return await balancedOnnxCounter.countSyllables(text);
    } catch (error) {
      console.error('Error using Balanced ONNX syllable counter, falling back to rule-based:', error);
      return countSyllablesRuleBased(text);
    }
  }"""
        
        # Find the position to insert the case
        simple_case_pos = updated_content.find("else if (currentCounterType === SyllableCounterType.SIMPLE_ONNX && simpleOnnxInitialized) {")
        if simple_case_pos != -1:
            # Find the end of the case
            simple_case_end = updated_content.find("}", simple_case_pos)
            if simple_case_end != -1:
                # Find the next line after the case
                next_line_pos = updated_content.find("\n", simple_case_end)
                if next_line_pos != -1:
                    # Insert after the case
                    updated_content = updated_content[:next_line_pos] + balanced_case + updated_content[next_line_pos:]
    
    # Write the updated file
    with open(counter_switch_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {counter_switch_path}")


if __name__ == "__main__":
    update_web_app()
