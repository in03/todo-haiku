// Simple ONNX-based syllable counter implementation
// NOTE: This implementation is currently disabled for performance reasons.
// The decision tree counter is used instead.
import { InferenceSession, Tensor } from 'onnxruntime-web';

// Define interface for model metadata
interface ModelMetadata {
  vocab_size: number;
  embed_dim: number;
  hidden_dim: number;
  max_word_len: number;
}

// Define interface for character vocabulary
interface CharVocabulary {
  [key: string]: number;
}

// Class for ONNX-based syllable counting
export class SimpleOnnxSyllableCounter {
  private session: InferenceSession | null = null;
  private charToIdx: CharVocabulary | null = null;
  private metadata: ModelMetadata | null = null;
  private maxWordLen: number = 20;
  private isLoaded: boolean = false;
  private isLoading: boolean = false;
  private loadError: Error | null = null;

  // Initialize the model
  async init(
    modelUrl: string = './models/simple_syllable_model.onnx',
    vocabUrl: string = './models/simple_char_vocab.json',
    metadataUrl: string = './models/simple_model_metadata.json'
  ): Promise<void> {
    if (this.isLoaded || this.isLoading) {
      return;
    }

    this.isLoading = true;

    try {
      // Create proper URLs
      const baseUrl = window.location.origin;
      const fullModelUrl = new URL(modelUrl, baseUrl).href;
      const fullVocabUrl = new URL(vocabUrl, baseUrl).href;
      const fullMetadataUrl = new URL(metadataUrl, baseUrl).href;

      // Load the ONNX model
      this.session = await InferenceSession.create(fullModelUrl);

      // Load the character vocabulary
      const vocabResponse = await fetch(fullVocabUrl);
      this.charToIdx = await vocabResponse.json();

      // Load the model metadata
      const metadataResponse = await fetch(fullMetadataUrl);
      this.metadata = await metadataResponse.json();
      this.maxWordLen = this.metadata.max_word_len;

      this.isLoaded = true;
    } catch (error) {
      this.loadError = error as Error;
      console.error('Error loading simple ONNX syllable counter:', error);
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

    // Convert word to character indices
    const charIds = new Array(this.maxWordLen).fill(0);
    for (let i = 0; i < Math.min(wordLower.length, this.maxWordLen); i++) {
      const char = wordLower[i];
      charIds[i] = this.charToIdx[char] || 0; // Use padding token (0) if character not in vocabulary
    }

    // Create input tensor
    const inputs = {
      'input': new Tensor('int64', new BigInt64Array(charIds.map(n => BigInt(n))), [1, this.maxWordLen])
    };

    // Run inference
    const outputMap = await this.session.run(inputs);
    const output = outputMap['output'];
    const prediction = output.data as Float32Array;

    // Round to nearest integer (since this is a regression model)
    return Math.round(prediction[0]);
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
let instance: SimpleOnnxSyllableCounter | null = null;

// Get the singleton instance
export function getSimpleOnnxSyllableCounter(): SimpleOnnxSyllableCounter {
  if (!instance) {
    instance = new SimpleOnnxSyllableCounter();
  }
  return instance;
}

