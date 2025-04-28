"""
Update the web application to use the character-level LSTM syllable counter model.
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
    Update the web application to use the character-level LSTM syllable counter model.
    
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
        os.path.join(model_dir, "syllable_char_lstm.onnx"),
        os.path.join(public_dir, "syllable_char_lstm.onnx")
    )
    print(f"Copied syllable_char_lstm.onnx")
    
    # Copy the vocabulary
    shutil.copy(
        os.path.join(model_dir, "char_vocab_lstm.json"),
        os.path.join(public_dir, "char_vocab_lstm.json")
    )
    print(f"Copied char_vocab_lstm.json")
    
    # Copy the metadata
    shutil.copy(
        os.path.join(model_dir, "model_metadata_char_lstm.json"),
        os.path.join(public_dir, "model_metadata_char_lstm.json")
    )
    print(f"Copied model_metadata_char_lstm.json")
    
    print("Model files copied successfully.")
    
    # Update the TypeScript files if requested
    if update_ts_files:
        update_typescript_files()


def update_typescript_files():
    """
    Update the TypeScript files to use the character-level LSTM syllable counter model.
    """
    # Define the files to update
    files_to_update = [
        "../src/utils/syllable-counter-switch.ts",
        "../src/utils/char-lstm-syllable-counter.ts"
    ]
    
    # Create the character-level LSTM syllable counter implementation
    create_char_lstm_counter_file()
    
    # Update the syllable counter switch
    update_counter_switch()
    
    print("TypeScript files updated successfully.")


def create_char_lstm_counter_file():
    """
    Create the character-level LSTM syllable counter implementation.
    """
    char_lstm_counter_path = "../src/utils/char-lstm-syllable-counter.ts"
    
    # Check if the file already exists
    if os.path.exists(char_lstm_counter_path):
        print(f"File {char_lstm_counter_path} already exists. Skipping creation.")
        return
    
    # Create the file
    with open(char_lstm_counter_path, "w") as f:
        f.write("""// Character-level LSTM syllable counter implementation
import { InferenceSession, Tensor } from 'onnxruntime-web';

// Define interface for model metadata
interface ModelMetadata {
  vocab_size: number;
  embedding_dim: number;
  hidden_dim: number;
  n_layers: number;
  dropout: number;
  test_accuracy: number;
}

// Define interface for character vocabulary
interface CharVocabulary {
  [key: string]: number;
}

// Class for character-level LSTM syllable counting
export class CharLstmSyllableCounter {
  private session: InferenceSession | null = null;
  private charToIdx: CharVocabulary | null = null;
  private metadata: ModelMetadata | null = null;
  private maxWordLen: number = 30;
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
      const metadataResponse = await fetch('/models/model_metadata_char_lstm.json');
      this.metadata = await metadataResponse.json();

      // Load character vocabulary
      const vocabResponse = await fetch('/models/char_vocab_lstm.json');
      this.charToIdx = await vocabResponse.json();

      // Create ONNX session
      this.session = await InferenceSession.create('/models/syllable_char_lstm.onnx');
      
      this.isLoaded = true;
      console.log('Character-level LSTM syllable counter model loaded successfully');
    } catch (error) {
      this.loadError = error as Error;
      console.error('Failed to load character-level LSTM syllable counter model:', error);
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
    
    // Convert word to character indices
    const charIds = new Array(this.maxWordLen).fill(0);
    for (let i = 0; i < Math.min(wordLower.length, this.maxWordLen); i++) {
      const char = wordLower[i];
      charIds[i] = this.charToIdx[char] || this.charToIdx['<unk>'] || 1; // Use <unk> token if character not in vocabulary
    }

    // Create input tensors
    const inputs = {
      'chars': new Tensor('int64', new BigInt64Array(charIds.map(n => BigInt(n))), [1, this.maxWordLen]),
      'lengths': new Tensor('int64', new BigInt64Array([BigInt(wordLower.length)]), [1])
    };

    // Run inference
    const outputMap = await this.session.run(inputs);
    const output = outputMap['bio_tags'];
    const predictions = output.data as Float32Array;

    // Get the predicted BIO tags
    const bioTags = [];
    for (let i = 0; i < wordLower.length; i++) {
      const offset = i * 3; // 3 classes: B, I, O
      const maxIdx = this.argmax([predictions[offset], predictions[offset + 1], predictions[offset + 2]]);
      bioTags.push(maxIdx);
    }

    // Count syllables from BIO tags (count 'B' tags)
    const syllableCount = bioTags.filter(tag => tag === 2).length;
    
    // Ensure at least one syllable
    return Math.max(1, syllableCount);
  }

  // Helper function to find the index of the maximum value in an array
  private argmax(array: number[]): number {
    return array.indexOf(Math.max(...array));
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
let instance: CharLstmSyllableCounter | null = null;

// Get the singleton instance
export function getCharLstmSyllableCounter(): CharLstmSyllableCounter {
  if (!instance) {
    instance = new CharLstmSyllableCounter();
  }
  return instance;
}
""")
    
    print(f"Created {char_lstm_counter_path}")


def update_counter_switch():
    """
    Update the syllable counter switch to include the character-level LSTM model.
    """
    counter_switch_path = "../src/utils/syllable-counter-switch.ts"
    
    # Check if the file exists
    if not os.path.exists(counter_switch_path):
        print(f"File {counter_switch_path} does not exist. Skipping update.")
        return
    
    # Read the file
    with open(counter_switch_path, "r") as f:
        content = f.read()
    
    # Check if the character-level LSTM model is already included
    if "CHAR_LSTM" in content:
        print(f"Character-level LSTM model already included in {counter_switch_path}. Skipping update.")
        return
    
    # Update the file
    updated_content = content.replace(
        "export enum SyllableCounterType {",
        "export enum SyllableCounterType {\n  CHAR_LSTM = 'CHAR_LSTM',"
    )
    
    # Add import for character-level LSTM model
    updated_content = updated_content.replace(
        "import { getCmuOnnxSyllableCounter } from './cmu-syllable-counter-onnx';",
        "import { getCmuOnnxSyllableCounter } from './cmu-syllable-counter-onnx';\nimport { getCharLstmSyllableCounter } from './char-lstm-syllable-counter';"
    )
    
    # Add initialization for character-level LSTM model
    if "let charLstmInitialized = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitialized = false;",
            "let cmuOnnxInitialized = false;\nlet charLstmInitialized = false;"
        )
    
    if "let charLstmInitializing = false;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitializing = false;",
            "let cmuOnnxInitializing = false;\nlet charLstmInitializing = false;"
        )
    
    if "let charLstmInitError: Error | null = null;" not in updated_content:
        updated_content = updated_content.replace(
            "let cmuOnnxInitError: Error | null = null;",
            "let cmuOnnxInitError: Error | null = null;\nlet charLstmInitError: Error | null = null;"
        )
    
    # Add initialization function for character-level LSTM model
    if "export async function initCharLstmModel(): Promise<void> {" not in updated_content:
        init_function = """
// Initialize the Character-level LSTM model
export async function initCharLstmModel(): Promise<void> {
  if (charLstmInitialized || charLstmInitializing) {
    return;
  }

  charLstmInitializing = true;
  try {
    const charLstmCounter = getCharLstmSyllableCounter();
    await charLstmCounter.init();
    charLstmInitialized = true;
    charLstmInitError = null;
  } catch (error) {
    charLstmInitError = error as Error;
    console.error('Failed to initialize Character-level LSTM model:', error);
    // Fall back to rule-based counter
    currentCounterType = SyllableCounterType.RULE_BASED;
  } finally {
    charLstmInitializing = false;
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
    if "else if (currentCounterType === SyllableCounterType.CHAR_LSTM && charLstmInitialized) {" not in updated_content:
        char_lstm_case = """
  else if (currentCounterType === SyllableCounterType.CHAR_LSTM && charLstmInitialized) {
    try {
      const charLstmCounter = getCharLstmSyllableCounter();
      return await charLstmCounter.countSyllables(text);
    } catch (error) {
      console.error('Error using Character-level LSTM syllable counter, falling back to rule-based:', error);
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
                    updated_content = updated_content[:next_line_pos] + char_lstm_case + updated_content[next_line_pos:]
    
    # Write the updated file
    with open(counter_switch_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {counter_switch_path}")


if __name__ == "__main__":
    update_web_app()
