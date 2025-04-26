import { component$, useSignal, useStore, useVisibleTask$ } from '@builder.io/qwik';
import { getAllTodos, updateTodo, deleteTodo } from '../services/yjs-sync';

export default component$(() => {
  const todos = useStore<any[]>([]);
  const filter = useSignal<'all' | 'active' | 'completed'>('all');

  // Load todos on component mount
  useVisibleTask$(({ track }) => {
    // Re-run when filter changes
    track(() => filter.value);

    try {
      // Get todos from Y.js
      const allTodos = getAllTodos();

      // Also try to get todos from localStorage as fallback
      let localStorageTodos = [];
      if (typeof window !== 'undefined' && window.localStorage) {
        try {
          localStorageTodos = JSON.parse(localStorage.getItem('todos') || '[]');
        } catch (e) {
          console.error('Error reading from localStorage:', e);
        }
      }

      // Combine todos from both sources (prefer Y.js)
      const combinedTodos = [...allTodos];

      // Add localStorage todos that aren't already in the Y.js todos
      const yJsIds = new Set(allTodos.map(todo => todo.id));
      for (const todo of localStorageTodos) {
        if (!yJsIds.has(todo.id)) {
          combinedTodos.push(todo);
        }
      }

      // Apply filter
      if (filter.value === 'active') {
        todos.length = 0;
        todos.push(...combinedTodos.filter(todo => !todo.is_completed));
      } else if (filter.value === 'completed') {
        todos.length = 0;
        todos.push(...combinedTodos.filter(todo => todo.is_completed));
      } else {
        todos.length = 0;
        todos.push(...combinedTodos);
      }

      // Sort by creation date (newest first)
      todos.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      console.log('Loaded todos:', todos.length);
    } catch (error) {
      console.error('Error loading todos:', error);
    }
  });

  // Toggle todo completion status
  const toggleTodo = (id: string, isCompleted: boolean) => {
    updateTodo(id, { is_completed: !isCompleted });

    // Update local state
    const index = todos.findIndex(todo => todo.id === id);
    if (index !== -1) {
      todos[index].is_completed = !isCompleted;
    }
  };

  // Delete a todo
  const removeTodo = (id: string) => {
    deleteTodo(id);

    // Update local state
    const index = todos.findIndex(todo => todo.id === id);
    if (index !== -1) {
      todos.splice(index, 1);
    }
  };

  return (
    <div class="zen-container">
      <div class="mb-6">
        <h2 class="text-2xl font-semibold mb-4">Your Tasks</h2>

        <div class="flex space-x-2 mb-6">
          <button
            onClick$={() => filter.value = 'all'}
            class={`px-3 py-1 rounded-md text-sm ${
              filter.value === 'all'
                ? 'bg-accent text-accent-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            All
          </button>
          <button
            onClick$={() => filter.value = 'active'}
            class={`px-3 py-1 rounded-md text-sm ${
              filter.value === 'active'
                ? 'bg-accent text-accent-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            Active
          </button>
          <button
            onClick$={() => filter.value = 'completed'}
            class={`px-3 py-1 rounded-md text-sm ${
              filter.value === 'completed'
                ? 'bg-accent text-accent-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            Completed
          </button>
        </div>

        {todos.length === 0 ? (
          <div class="text-center py-8 text-muted-foreground">
            <p>No tasks found. Create a new one!</p>
          </div>
        ) : (
          <div>
            {todos.map(todo => (
              <div key={todo.id} class="haiku-card">
                <div class="flex justify-between items-start mb-2">
                  <h3 class={`text-lg font-medium ${todo.is_completed ? 'line-through text-muted-foreground' : ''}`}>
                    {todo.title}
                  </h3>
                  <div class="flex space-x-2">
                    <button
                      onClick$={() => toggleTodo(todo.id, todo.is_completed)}
                      class={`p-1 rounded-md ${
                        todo.is_completed
                          ? 'text-green-500 hover:text-green-400'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                      aria-label={todo.is_completed ? 'Mark as incomplete' : 'Mark as complete'}
                    >
                      {todo.is_completed ? '✓' : '○'}
                    </button>
                    <button
                      onClick$={() => removeTodo(todo.id)}
                      class="p-1 text-destructive hover:text-destructive/80 rounded-md"
                      aria-label="Delete task"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div class="haiku-text whitespace-pre-line">
                  {todo.content}
                </div>
                <div class="text-xs text-muted-foreground mt-2">
                  {new Date(todo.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});
