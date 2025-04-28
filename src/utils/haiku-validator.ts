// Import our simple rule-based syllable counter
import { countSyllables } from './syllable-counter';

// Define the haiku pattern (5-7-5 syllables)
const HAIKU_PATTERN = [5, 7, 5];

// Validate if text follows haiku pattern
export function validateHaiku(text: string): {
  isValid: boolean;
  syllableCounts: number[];
  feedback: string;
} {
  // Split text into lines and remove empty lines
  const lines = text.split('\n').filter(line => line.trim() !== '');

  // Check if we have exactly 3 lines
  if (lines.length !== 3) {
    // Count syllables for each line
    const syllableCounts = lines.map(line => countSyllables(line));

    return {
      isValid: false,
      syllableCounts,
      feedback: "A haiku needs exactly three lines."
    };
  }

  // Count syllables for each line
  const syllableCounts = lines.map(line => countSyllables(line));

  // Check if syllable counts match the haiku pattern
  const isValid = syllableCounts.every((count, index) => count === HAIKU_PATTERN[index]);

  // Generate feedback based on validation
  let feedback = "";

  if (isValid) {
    const feedbacks = [
      "Perfect haiku! You're a natural poet.",
      "Wow, a real poet!",
      "Your haiku flows like a gentle stream.",
      "Basho would be proud!",
      "A moment captured in seventeen syllables."
    ];
    feedback = feedbacks[Math.floor(Math.random() * feedbacks.length)];
  } else {
    // Check which lines don't match the pattern
    const invalidLines = syllableCounts.map((count, index) => {
      if (count !== HAIKU_PATTERN[index]) {
        const diff = HAIKU_PATTERN[index] - count;
        return {
          line: index + 1,
          expected: HAIKU_PATTERN[index],
          actual: count,
          diff
        };
      }
      return null;
    }).filter(Boolean);

    if (invalidLines.length === 1) {
      const invalidLine = invalidLines[0];
      if (invalidLine) {
        const { line, diff } = invalidLine;
        if (diff > 0) {
          feedback = `Line ${line} needs ${diff} more syllable${diff !== 1 ? 's' : ''}.`;
        } else {
          feedback = `Line ${line} has ${Math.abs(diff)} too many syllable${Math.abs(diff) !== 1 ? 's' : ''}.`;
        }
      }
    } else {
      const feedbacks = [
        "Not quite a haiku, but it's a start!",
        "Oops, missing a few syllables.",
        "It's not the best, but oh well.",
        "Almost there! Check your syllable count.",
        "A valiant attempt at poetry."
      ];
      feedback = feedbacks[Math.floor(Math.random() * feedbacks.length)];
    }
  }

  return { isValid, syllableCounts, feedback };
}

// Generate a random haiku template to inspire users
export function generateHaikuTemplate(): string {
  const templates = [
    "Morning sun rises\nDew drops glisten on green leaves\nA new day begins",
    "Typing on keyboard\nThoughts flow into characters\nTasks become haikus",
    "Mountain silhouette\nShadows dance across the lake\nPeace in solitude",
    "Deadline approaching\nFingers race across the keys\nWork becomes a blur",
    "Empty task list waits\nIdeas form in my mind\nTime to write them down"
  ];

  return templates[Math.floor(Math.random() * templates.length)];
}
