import { component$, useSignal, useStore, $ } from '@builder.io/qwik';
import { validateHaiku, generateHaikuTemplate } from '../utils/haiku-validator';
import { v4 as uuidv4 } from 'uuid';
import { addTodo } from '../services/yjs-sync';

export default component$(() => {
  const content = useSignal('');
  const title = useSignal('');
  const validation = useStore({
    isValid: false,
    syllableCounts: [0, 0, 0],
    feedback: ''
  });

  // Validate haiku as user types
  const validateContent = $(() => {
    const result = validateHaiku(content.value);
    validation.isValid = result.isValid;
    validation.syllableCounts = result.syllableCounts;
    validation.feedback = result.feedback;
  });

  // Generate a random haiku template
  const generateTemplate = $(() => {
    content.value = generateHaikuTemplate();
    validateContent();
  });

  // Submit the todo
  const submitTodo = $(() => {
    if (!validation.isValid || !title.value.trim()) {
      return;
    }

    const newTodo = {
      id: uuidv4(),
      title: title.value,
      content: content.value,
      is_completed: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      user_id: 'local' // This will be replaced with the actual user ID when synced
    };

    addTodo(newTodo);

    // Reset form
    title.value = '';
    content.value = '';
    validation.isValid = false;
    validation.syllableCounts = [0, 0, 0];
    validation.feedback = '';
  });

  return (
    <div class="zen-container">
      <div class="mb-8">
        <h2 class="text-2xl font-semibold mb-4">Create a New Task</h2>

        <div class="mb-4">
          <label class="block text-sm font-medium mb-1" for="title">
            Task Title
          </label>
          <input
            id="title"
            type="text"
            value={title.value}
            onInput$={(e: any) => title.value = e.target.value}
            class="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-1 focus:ring-accent"
            placeholder="Enter a title for your task"
          />
        </div>

        <div class="mb-4">
          <div class="flex justify-between items-center mb-1">
            <label class="block text-sm font-medium" for="content">
              Task Description (as a Haiku)
            </label>
            <button
              onClick$={generateTemplate}
              class="text-xs text-accent hover:text-accent/80"
            >
              Generate Template
            </button>
          </div>
          <textarea
            id="content"
            value={content.value}
            onInput$={(e: any) => {
              content.value = e.target.value;
              validateContent();
            }}
            class="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-1 focus:ring-accent font-serif h-32"
            placeholder="Write your task as a haiku (5-7-5 syllables)"
          />
        </div>

        <div class="mb-6">
          <div class="flex justify-between text-xs text-muted-foreground">
            <span>Line 1: {validation.syllableCounts[0] || 0}/5 syllables</span>
            <span>Line 2: {validation.syllableCounts[1] || 0}/7 syllables</span>
            <span>Line 3: {validation.syllableCounts[2] || 0}/5 syllables</span>
          </div>
          <p class={`text-sm mt-1 ${validation.isValid ? 'text-green-500' : 'text-amber-500'}`}>
            {validation.feedback}
          </p>
        </div>

        <button
          onClick$={submitTodo}
          disabled={!validation.isValid || !title.value.trim()}
          class="w-full py-2 px-4 bg-accent text-accent-foreground rounded-md hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Create Task
        </button>
      </div>
    </div>
  );
});
