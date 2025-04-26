// A simple syllable counter for English text
// This is a simplified version and may not be 100% accurate for all words

// Define special cases and exceptions
const specialCases: Record<string, number> = {
  // Common words with irregular syllable counts
  'every': 2,
  'different': 3,
  'beautiful': 3,
  'interesting': 3,
  'experience': 4,
  'favorite': 3,
  'family': 3,
  'evening': 2,
  'area': 3,
  'hour': 1,
  'fire': 1,
  'poem': 2,
  'poems': 2,
  'quiet': 2,
  'science': 2,
  'society': 3,
  'though': 1,
  'through': 1,
  'throughout': 2,
  'wednesday': 3,
  'forest': 2,
  'poetry': 3,
  'haiku': 2,
  'syllable': 3,
  'syllables': 3,
};

// Count syllables in a word
function countSyllablesInWord(word: string): number {
  // Remove punctuation and convert to lowercase
  word = word.toLowerCase().replace(/[.,;:!?()'"]/g, '');
  
  // Check for special cases
  if (specialCases[word] !== undefined) {
    return specialCases[word];
  }
  
  // Count syllables based on vowel groups
  const vowels = 'aeiouy';
  let count = 0;
  let prevIsVowel = false;
  
  // Handle specific patterns
  if (word.length > 0 && vowels.includes(word[0])) {
    count = 1;
    prevIsVowel = true;
  }
  
  for (let i = 1; i < word.length; i++) {
    const isVowel = vowels.includes(word[i]);
    
    if (isVowel && !prevIsVowel) {
      count++;
    }
    
    prevIsVowel = isVowel;
  }
  
  // Handle silent e at the end
  if (word.length > 2 && word.endsWith('e') && !vowels.includes(word[word.length - 2])) {
    count = Math.max(1, count - 1);
  }
  
  // Handle words ending with 'le' where the 'l' is preceded by a consonant
  if (word.length > 2 && word.endsWith('le') && !vowels.includes(word[word.length - 3])) {
    count++;
  }
  
  // Handle words ending with 'ed'
  if (word.length > 2 && word.endsWith('ed')) {
    // Only count as a syllable if preceded by t or d
    const prevChar = word[word.length - 3];
    if (prevChar !== 't' && prevChar !== 'd') {
      count = Math.max(1, count - 1);
    }
  }
  
  // Ensure at least one syllable
  return Math.max(1, count);
}

// Count syllables in a line of text
export function countSyllables(text: string): number {
  if (!text) return 0;
  
  // Split text into words
  const words = text.split(/\s+/).filter(word => word.length > 0);
  
  // Count syllables in each word and sum them
  return words.reduce((total, word) => total + countSyllablesInWord(word), 0);
}
