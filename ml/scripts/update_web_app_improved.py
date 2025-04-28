"""
Update the web application to use the improved CMU-based syllable counter model.
"""
import os
import shutil
import json
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def update_web_app(
    model_dir="models",
    public_dir="../public/models",
    update_ts_files=True
):
    """
    Update the web application to use the improved CMU-based syllable counter model.
    
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
        os.path.join(model_dir, "syllable_counter_cmu_improved.onnx"),
        os.path.join(public_dir, "syllable_counter_cmu_improved.onnx")
    )
    print(f"Copied syllable_counter_cmu_improved.onnx")
    
    # Copy the vocabulary
    shutil.copy(
        os.path.join(model_dir, "char_vocab_cmu_improved.json"),
        os.path.join(public_dir, "char_vocab_cmu_improved.json")
    )
    print(f"Copied char_vocab_cmu_improved.json")
    
    # Copy the metadata
    shutil.copy(
        os.path.join(model_dir, "model_metadata_cmu_improved.json"),
        os.path.join(public_dir, "model_metadata_cmu_improved.json")
    )
    print(f"Copied model_metadata_cmu_improved.json")
    
    print("Model files copied successfully.")
    
    # Update the TypeScript files if requested
    if update_ts_files:
        update_typescript_files()


def update_typescript_files():
    """
    Update the TypeScript files to use the improved CMU-based syllable counter model.
    """
    # Define the files to update
    files_to_update = [
        "../src/utils/syllable-counter-switch.ts",
        "../src/utils/improved-syllable-counter-onnx.ts"
    ]
    
    # Create the improved syllable counter ONNX implementation
    create_improved_counter_file()
    
    # Update the syllable counter switch
    update_counter_switch()
    
    print("TypeScript files updated successfully.")


def create_improved_counter_file():
    """
    Create the improved syllable counter ONNX implementation.
    """
    improved_counter_path = "../src/utils/improved-syllable-counter-onnx.ts"
    
    # Check if the file already exists
    if os.path.exists(improved_counter_path):
        print(f"File {improved_counter_path} already exists. Skipping creation.")
        return
    
    # Create the file
    with open(improved_counter_path, "w") as f:
        f.write("""// Improved CMU-based ONNX syllable counter implementation
import { InferenceSession, Tensor } from 'onnxruntime-web';

// Define interface for model metadata
interface ModelMetadata {
  max_word_len: number;
  vocab_size: number;
  embedding_dim: number;
  hidden_dim: number;
  num_features: number;
  test_accuracy: number;
  bias_correction: number;
}

// Define interface for character vocabulary
interface CharVocabulary {
  [key: string]: number;
}

// Class for improved CMU-based ONNX syllable counting
export class ImprovedOnnxSyllableCounter {
  private session: InferenceSession | null = null;
  private charToIdx: CharVocabulary | null = null;
  private metadata: ModelMetadata | null = null;
  private maxWordLen: number = 20;
  private biasCorrection: number = 0;
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
      const metadataResponse = await fetch('/models/model_metadata_cmu_improved.json');
      this.metadata = await metadataResponse.json();
      this.maxWordLen = this.metadata.max_word_len;
      this.biasCorrection = this.metadata.bias_correction || 0;

      // Load character vocabulary
      const vocabResponse = await fetch('/models/char_vocab_cmu_improved.json');
      this.charToIdx = await vocabResponse.json();

      // Create ONNX session
      this.session = await InferenceSession.create('/models/syllable_counter_cmu_improved.onnx');
      
      this.isLoaded = true;
      console.log('Improved syllable counter model loaded successfully');
      console.log(`Bias correction: ${this.biasCorrection}`);
    } catch (error) {
      this.loadError = error as Error;
      console.error('Failed to load improved syllable counter model:', error);
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

    // Apply bias correction
    const correctedPrediction = Math.max(1, maxIndex - this.biasCorrection);
    return correctedPrediction;
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
let instance: ImprovedOnnxSyllableCounter | null = null;

// Get the singleton instance
export function getImprovedOnnxSyllableCounter(): ImprovedOnnxSyllableCounter {
  if (!instance) {
    instance = new ImprovedOnnxSyllableCounter();
  }
  return instance;
}
""")
    
    print(f"Created {improved_counter_path}")


def update_counter_switch():
    """
    Update the syllable counter switch to include the improved model.
    """
    counter_switch_path = "../src/utils/syllable-counter-switch.ts"
    
    # Check if the file exists
    if not os.path.exists(counter_switch_path):
        print(f"File {counter_switch_path} does not exist. Skipping update.")
        return
    
    # Read the file
    with open(counter_switch_path, "r") as f:
        content = f.read()
    
    # Check if the improved model is already included
    if "IMPROVED_ONNX" in content:
        print(f"Improved model already included in {counter_switch_path}. Skipping update.")
        return
    
    # Update the file
    updated_content = content.replace(
        "export enum SyllableCounterType {",
        "export enum SyllableCounterType {\n  IMPROVED_ONNX = 'IMPROVED_ONNX',"
    )
    
    # Add import for improved model
    updated_content = updated_content.replace(
        "import { getCmuOnnxSyllableCounter } from './cmu-syllable-counter-onnx';",
        "import { getCmuOnnxSyllableCounter } from './cmu-syllable-counter-onnx';\nimport { getImprovedOnnxSyllableCounter } from './improved-syllable-counter-onnx';"
    )
    
    # Add initialization for improved model
    if "let improvedOnnxInitialized = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitialized = false;",
            "let cmuOnnxInitialized = false;\nlet improvedOnnxInitialized = false;"
        )
    
    if "let improvedOnnxInitializing = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitializing = false;",
            "let cmuOnnxInitializing = false;\nlet improvedOnnxInitializing = false;"
        )
    
    if "let improvedOnnxInitError: Error | null = null;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitError: Error | null = null;",
            "let cmuOnnxInitError: Error | null = null;\nlet improvedOnnxInitError: Error | null = null;"
        )
    
    # Add initialization function for improved model
    if "export async function initImprovedOnnxModel(): Promise<void> {" not in updated_content:
        init_function = """
// Initialize the Improved ONNX model
export async function initImprovedOnnxModel(): Promise<void> {
  if (improvedOnnxInitialized || improvedOnnxInitializing) {
    return;
  }

  improvedOnnxInitializing = true;
  try {
    const improvedOnnxCounter = getImprovedOnnxSyllableCounter();
    await improvedOnnxCounter.init();
    improvedOnnxInitialized = true;
    improvedOnnxInitError = null;
  } catch (error) {
    improvedOnnxInitError = error as Error;
    console.error('Failed to initialize Improved ONNX model:', error);
    // Fall back to rule-based counter
    currentCounterType = SyllableCounterType.RULE_BASED;
  } finally {
    improvedOnnxInitializing = false;
  }
}"""
        
        # Find the position to insert the function
        init_cmu_pos = updated_content.find("export async function initCmuOnnxModel()")
        if init_cmu_pos != -1:
            # Find the end of the function
            init_cmu_end = updated_content.find("}", init_cmu_pos)
            if init_cmu_end != -1:
                # Insert after the function
                updated_content = updated_content[:init_cmu_end+1] + init_function + updated_content[init_cmu_end+1:]
    
    # Update the countSyllables function
    if "else if (currentCounterType === SyllableCounterType.IMPROVED_ONNX && improvedOnnxInitialized) {" not in updated_content:
        improved_case = """
  else if (currentCounterType === SyllableCounterType.IMPROVED_ONNX && improvedOnnxInitialized) {
    try {
      const improvedOnnxCounter = getImprovedOnnxSyllableCounter();
      return await improvedOnnxCounter.countSyllables(text);
    } catch (error) {
      console.error('Error using Improved ONNX syllable counter, falling back to rule-based:', error);
      return countSyllablesRuleBased(text);
    }
  }"""
        
        # Find the position to insert the case
        cmu_case_pos = updated_content.find("else if (currentCounterType === SyllableCounterType.CMU_ONNX && cmuOnnxInitialized) {")
        if cmu_case_pos != -1:
            # Find the end of the case
            cmu_case_end = updated_content.find("}", cmu_case_pos)
            if cmu_case_end != -1:
                # Find the next line after the case
                next_line_pos = updated_content.find("\n", cmu_case_end)
                if next_line_pos != -1:
                    # Insert after the case
                    updated_content = updated_content[:next_line_pos] + improved_case + updated_content[next_line_pos:]
    
    # Write the updated file
    with open(counter_switch_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {counter_switch_path}")


if __name__ == "__main__":
    update_web_app()
