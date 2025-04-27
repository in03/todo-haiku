import { Task } from '~/types/task';

/**
 * Simple fuzzy search implementation
 * @param text The text to search in
 * @param query The search query
 * @returns True if the query matches the text, false otherwise
 */
export function fuzzySearch(text: string, query: string): boolean {
  if (!query) return true;

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();

  // Simple implementation: check if all characters in the query appear in order in the text
  let textIndex = 0;

  for (let queryIndex = 0; queryIndex < lowerQuery.length; queryIndex++) {
    const queryChar = lowerQuery[queryIndex];

    // Skip spaces in the query
    if (queryChar === ' ') continue;

    // Find the next occurrence of the current query character
    const charIndex = lowerText.indexOf(queryChar, textIndex);

    // If the character is not found, the search fails
    if (charIndex === -1) return false;

    // Move the text index forward
    textIndex = charIndex + 1;
  }

  return true;
}

/**
 * Filter tasks based on a search term
 * @param tasks The tasks to filter
 * @param searchTerm The search term
 * @returns Filtered tasks
 */
export function filterTasksBySearchTerm(tasks: Task[], searchTerm: string): Task[] {
  console.log('filterTasksBySearchTerm called with', tasks.length, 'tasks and search term:', searchTerm);

  if (!searchTerm) {
    console.log('No search term, returning all tasks');
    return tasks;
  }

  const filteredTasks = tasks.filter(task => {
    // Search in title and content
    return (
      fuzzySearch(task.title, searchTerm) ||
      fuzzySearch(task.content, searchTerm)
    );
  });

  console.log('Filtered to', filteredTasks.length, 'tasks');
  return filteredTasks;
}

/**
 * Sort tasks based on the specified sort option
 * @param tasks The tasks to sort
 * @param sortOption The sort option
 * @returns Sorted tasks
 */
export function sortTasks(tasks: Task[], sortOption: string): Task[] {
  console.log('sortTasks called with', tasks.length, 'tasks and sort option:', sortOption);

  const tasksCopy = [...tasks];

  switch (sortOption) {
    case 'title':
      return tasksCopy.sort((a, b) => a.title.localeCompare(b.title));

    case 'date':
      return tasksCopy.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

    case 'mood':
      // Simple mood sorting - just a placeholder implementation
      // In a real app, you might analyze the content for sentiment
      return tasksCopy.sort((a, b) => {
        // Count positive words as a simple proxy for mood
        const positiveWords = ['happy', 'joy', 'good', 'great', 'excellent', 'wonderful'];
        const aScore = positiveWords.reduce(
          (score, word) => score + (a.content.toLowerCase().includes(word) ? 1 : 0),
          0
        );
        const bScore = positiveWords.reduce(
          (score, word) => score + (b.content.toLowerCase().includes(word) ? 1 : 0),
          0
        );
        return bScore - aScore;
      });

    case 'custom':
    default:
      // Sort by sortOrder if available, otherwise by creation date
      const sorted = tasksCopy.sort((a, b) => {
        if (a.sortOrder !== undefined && b.sortOrder !== undefined) {
          return a.sortOrder - b.sortOrder;
        }
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      console.log('Sorted tasks (custom):', sorted.length);
      return sorted;
  }
}
