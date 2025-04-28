// Syllable counter switch utility
import { countSyllables as countSyllablesRuleBased } from './syllable-counter';
import { countSyllables as countSyllablesDecisionTree } from './decision-tree-syllable-counter';

// Enum for syllable counter types
export enum SyllableCounterType {
  RULE_BASED = 'rule-based',
  DECISION_TREE = 'decision-tree',
  // Keep these for backward compatibility, but they won't be used
  ONNX = 'onnx',
  SIMPLE_ONNX = 'simple-onnx'
}

// Default counter type - using decision tree as default now
let currentCounterType: SyllableCounterType = SyllableCounterType.DECISION_TREE;

// ONNX model initialization status - always false now
let onnxInitialized = false;
let onnxInitializing = false;
let onnxInitError: Error | null = new Error('ONNX models disabled for performance reasons');

// Simple ONNX model initialization status - always false now
let simpleOnnxInitialized = false;
let simpleOnnxInitializing = false;
let simpleOnnxInitError: Error | null = new Error('ONNX models disabled for performance reasons');

// Initialize the ONNX model - now a no-op function
export async function initOnnxModel(): Promise<void> {
  console.log('ONNX models disabled for performance reasons');
  return Promise.resolve();
}

// Initialize the Simple ONNX model - now a no-op function
export async function initSimpleOnnxModel(): Promise<void> {
  console.log('ONNX models disabled for performance reasons');
  return Promise.resolve();
}

// Set the current counter type
export function setSyllableCounterType(type: SyllableCounterType): void {
  // Redirect ONNX types to decision tree
  if (type === SyllableCounterType.ONNX || type === SyllableCounterType.SIMPLE_ONNX) {
    console.warn(`${type} model disabled for performance reasons, using decision tree instead`);
    currentCounterType = SyllableCounterType.DECISION_TREE;
  } else {
    currentCounterType = type;
  }
}

// Get the current counter type
export function getSyllableCounterType(): SyllableCounterType {
  return currentCounterType;
}

// Check if ONNX model is available - always false now
export function isOnnxModelAvailable(): boolean {
  return false;
}

// Check if Simple ONNX model is available - always false now
export function isSimpleOnnxModelAvailable(): boolean {
  return false;
}

// Get ONNX initialization status
export function getOnnxInitStatus(): {
  initialized: boolean;
  initializing: boolean;
  error: Error | null;
} {
  return {
    initialized: false,
    initializing: false,
    error: onnxInitError
  };
}

// Get Simple ONNX initialization status
export function getSimpleOnnxInitStatus(): {
  initialized: boolean;
  initializing: boolean;
  error: Error | null;
} {
  return {
    initialized: false,
    initializing: false,
    error: simpleOnnxInitError
  };
}

// Count syllables using the current counter
export async function countSyllables(text: string): Promise<number> {
  if (!text) return 0;

  if (currentCounterType === SyllableCounterType.DECISION_TREE) {
    try {
      return countSyllablesDecisionTree(text);
    } catch (error) {
      console.error('Error using Decision Tree syllable counter, falling back to rule-based:', error);
      return countSyllablesRuleBased(text);
    }
  } else {
    return countSyllablesRuleBased(text);
  }
}
