// ONNX-based syllable counter implementation
// NOTE: This implementation is currently disabled for performance reasons.
// The decision tree counter is used instead.
import { InferenceSession, Tensor, env } from 'onnxruntime-web';

// Configure ONNX runtime with more memory
// Use paths relative to the public directory
env.wasm.wasmPaths = {
  'ort-wasm-simd.wasm': '/node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': '/node_modules/onnxruntime-web/dist/ort-wasm-threaded.wasm',
  'ort-wasm-simd-threaded.wasm': '/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'
} as typeof env.wasm.wasmPaths;

// Configure WASM options
env.wasm.numThreads = 1; // Use single-threaded mode to reduce memory usage

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

// Class for ONNX-based syllable counting
export class OnnxSyllableCounter {
  private session: InferenceSession | null = null;
  private charToIdx: CharVocabulary | null = null;
  private metadata: ModelMetadata | null = null;
  private maxWordLen: number = 20;
  private isLoaded: boolean = false;
  private isLoading: boolean = false;
  private loadError: Error | null = null;

  // Initialize the model
  async init(
    modelUrl: string = './models/syllable_counter.onnx',
    vocabUrl: string = './models/char_vocab.json',
    metadataUrl: string = './models/model_metadata.json'
  ): Promise<void> {
    // Skip initialization during SSR
    if (typeof window === 'undefined') {
      console.log('Skipping ONNX model initialization during SSR');
      return;
    }

    if (this.isLoaded || this.isLoading) {
      return;
    }

    this.isLoading = true;

    try {
      // Check if files exist first
      console.log('Checking if model files exist...');

      // Create proper URLs - handle both client and server environments
      let baseUrl = '/';
      if (typeof window !== 'undefined') {
        baseUrl = window.location.origin;
      }

      // Remove leading './' from paths if present
      const cleanModelUrl = modelUrl.replace(/^\.\//, '');
      const cleanVocabUrl = vocabUrl.replace(/^\.\//, '');
      const cleanMetadataUrl = metadataUrl.replace(/^\.\//, '');

      const fullModelUrl = new URL(cleanModelUrl, baseUrl).href;
      const fullVocabUrl = new URL(cleanVocabUrl, baseUrl).href;
      const fullMetadataUrl = new URL(cleanMetadataUrl, baseUrl).href;

      // Check vocabulary file
      try {
        console.log(`Fetching vocabulary file from: ${fullVocabUrl}`);
        const vocabResponse = await fetch(fullVocabUrl);
        if (!vocabResponse.ok) {
          throw new Error(`Failed to fetch vocabulary file: ${vocabResponse.status} ${vocabResponse.statusText}`);
        }
        this.charToIdx = await vocabResponse.json();
        console.log('Character vocabulary loaded successfully');
      } catch (vocabError) {
        console.error('Error loading vocabulary:', vocabError);
        throw vocabError;
      }

      // Check metadata file
      try {
        console.log(`Fetching metadata file from: ${fullMetadataUrl}`);
        const metadataResponse = await fetch(fullMetadataUrl);
        if (!metadataResponse.ok) {
          throw new Error(`Failed to fetch metadata file: ${metadataResponse.status} ${metadataResponse.statusText}`);
        }
        this.metadata = await metadataResponse.json();
        this.maxWordLen = this.metadata.max_word_len;
        console.log('Model metadata loaded successfully');
      } catch (metadataError) {
        console.error('Error loading metadata:', metadataError);
        throw metadataError;
      }

      // Configure session options with minimal settings
      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: 'sequential'
      };

      // Load the ONNX model with options
      console.log(`Creating ONNX session with model from: ${fullModelUrl}`);
      console.log('Options:', options);

      // Try to fetch the model file first to check if it exists
      try {
        const modelResponse = await fetch(fullModelUrl);
        if (!modelResponse.ok) {
          throw new Error(`Failed to fetch model file: ${modelResponse.status} ${modelResponse.statusText}`);
        }
        console.log('Model file exists, proceeding with session creation');
      } catch (modelFetchError) {
        console.error('Error fetching model file:', modelFetchError);
        throw modelFetchError;
      }

      this.session = await InferenceSession.create(fullModelUrl, options);
      console.log('ONNX session created successfully');

      this.isLoaded = true;
      console.log('ONNX syllable counter initialized successfully');
    } catch (error) {
      this.loadError = error as Error;
      console.error('Error loading ONNX syllable counter:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // Check if the model is loaded
  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  // Get the load error if any
  getLoadError(): Error | null {
    return this.loadError;
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
      'word_chars': new Tensor('int64', new BigInt64Array(charIds.map(n => BigInt(n))), [1, this.maxWordLen]),
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
    const words = text.split(/\s+/).filter(word => word.length > 0);

    // Count syllables in each word and sum them
    let totalSyllables = 0;
    for (const word of words) {
      totalSyllables += await this.countSyllablesInWord(word);
    }

    return totalSyllables;
  }
}

// Singleton instance
let instance: OnnxSyllableCounter | null = null;

// Get the singleton instance
export function getOnnxSyllableCounter(): OnnxSyllableCounter {
  if (!instance) {
    instance = new OnnxSyllableCounter();
  }
  return instance;
}




