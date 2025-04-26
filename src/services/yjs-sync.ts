import * as Y from 'yjs';
import { supabase } from './supabase';
import { TaskStatus } from '~/types/task';

// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// Helper function to get valid task statuses
// This will automatically update if TaskStatus type changes
const getValidStatuses = (): string[] => {
  // This approach ensures that if you update the TaskStatus type in task.ts,
  // you only need to update this array in one place
  const validStatuses: TaskStatus[] = ['open', 'doing', 'done', 'blocked'];
  return validStatuses;
};

// Create variables that will be initialized in browser context
let ydoc;
let todosMap;
let indexeddbProvider;
let isSyncing = false;
let isOnline = false;

// Initialize Y.js only in browser context
if (isBrowser) {
  // Create a Y.js document
  ydoc = new Y.Doc();

  // Create a todos map in the shared document
  todosMap = ydoc.getMap('todos');

  // Set online status
  isOnline = navigator.onLine;

  // Set up IndexedDB persistence
  try {
    // Use a more browser-friendly approach to import
    Promise.resolve().then(async () => {
      try {
        const { IndexeddbPersistence } = await import('y-indexeddb');
        indexeddbProvider = new IndexeddbPersistence('todo-haiku-yjs', ydoc);

        // Initialize sync when the provider is synced
        indexeddbProvider.on('synced', () => {
          console.log('Loaded data from IndexedDB');

          if (isOnline) {
            syncWithSupabase();
          }
        });
      } catch (error) {
        console.error('Error initializing IndexedDB persistence:', error);
      }
    });
  } catch (error) {
    console.error('Error setting up Y.js:', error);
  }
} else {
  // Create dummy objects for SSR
  ydoc = { getMap: () => ({ set: () => {}, delete: () => {}, observe: () => {} }) };
  todosMap = { set: () => {}, delete: () => {}, observe: () => {}, entries: () => [] };
}

// Listen for online/offline events (only in browser)
if (isBrowser) {
  window.addEventListener('online', () => {
    isOnline = true;
    syncWithSupabase();
  });

  window.addEventListener('offline', () => {
    isOnline = false;
  });
}

// Function to sync local changes with Supabase
export async function syncWithSupabase() {
  if (!isOnline || isSyncing) return;

  if (!supabase) {
    console.error('Supabase client not initialized');
    return;
  }

  try {
    isSyncing = true;

    // Get current user
    const { data: sessionData } = await supabase.auth.getSession();
    const userId = sessionData.session?.user.id;

    if (!userId) {
      isSyncing = false;
      return;
    }

    // Get all todos from Supabase
    const { data: remoteTodos, error } = await supabase
      .from('todos')
      .select('*')
      .eq('user_id', userId);

    if (error) {
      console.error('Error fetching remote todos:', error);
      isSyncing = false;
      return;
    }

    // Get all local todos
    const localTodos = Array.from(todosMap.entries()).map(([id, value]) => ({
      id,
      ...value
    }));

    // Sync remote todos to local
    for (const remoteTodo of remoteTodos) {
      const localTodo = todosMap.get(remoteTodo.id);

      // If remote todo doesn't exist locally or has been updated more recently
      if (!localTodo || new Date(remoteTodo.updated_at) > new Date(localTodo.updated_at)) {
        // Ensure the task has a valid status
        const validStatuses = getValidStatuses();
        if (!remoteTodo.status || !validStatuses.includes(remoteTodo.status)) {
          remoteTodo.status = remoteTodo.is_completed ? 'done' : 'open';
        }

        todosMap.set(remoteTodo.id, remoteTodo);
      }
    }

    // Sync local todos to remote
    for (const [id, localTodo] of todosMap.entries()) {
      const remoteTodo = remoteTodos.find(todo => todo.id === id);

      // If local todo doesn't exist remotely or has been updated more recently
      if (!remoteTodo || new Date(localTodo.updated_at) > new Date(remoteTodo.updated_at)) {
        // Ensure the task has a valid status
        const validStatuses = getValidStatuses();
        const status = validStatuses.includes(localTodo.status)
          ? localTodo.status
          : (localTodo.is_completed ? 'done' : 'open');

        if (supabase) {
          await supabase
            .from('todos')
            .upsert({
              id,
              title: localTodo.title,
              content: localTodo.content,
              is_completed: localTodo.is_completed,
              status: status,
              user_id: userId,
              updated_at: localTodo.updated_at
            });
        }
      }
    }

    // Handle deleted todos
    const remoteIds = new Set(remoteTodos.map(todo => todo.id));
    const localIds = new Set(Array.from(todosMap.keys()));

    // Delete remote todos that don't exist locally
    for (const remoteId of remoteIds) {
      if (!localIds.has(remoteId) && supabase) {
        await supabase
          .from('todos')
          .delete()
          .eq('id', remoteId);
      }
    }

  } catch (error) {
    console.error('Error syncing with Supabase:', error);
  } finally {
    isSyncing = false;
  }
}

// Function to add a todo locally
export function addTodo(todo) {
  if (!isBrowser || !todosMap) {
    console.warn('Cannot add todo: browser environment or todosMap not available');
    return;
  }

  try {
    // Ensure the task has a valid status
    const validStatuses = getValidStatuses();
    if (!todo.status || !validStatuses.includes(todo.status)) {
      todo.status = todo.is_completed ? 'done' : 'open';
    }

    // Add to local storage for immediate feedback
    const todoWithTimestamp = {
      ...todo,
      updated_at: new Date().toISOString()
    };

    // Store in Y.js map
    todosMap.set(todo.id, todoWithTimestamp);

    // Also store in localStorage as fallback
    if (isBrowser && window.localStorage) {
      try {
        // Get existing todos
        const existingTodos = JSON.parse(localStorage.getItem('todos') || '[]');

        // Add new todo
        existingTodos.push(todoWithTimestamp);

        // Save back to localStorage
        localStorage.setItem('todos', JSON.stringify(existingTodos));
      } catch (e) {
        console.error('Error saving to localStorage:', e);
      }
    }

    // Sync with Supabase if online
    if (isOnline) {
      syncWithSupabase();
    }

    console.log('Todo added successfully:', todo.id);
  } catch (error) {
    console.error('Error adding todo:', error);
  }
}

// Function to update a todo locally
export function updateTodo(id, updates) {
  const todo = todosMap.get(id);

  if (todo) {
    // If updating status, also update is_completed for backward compatibility
    if (updates.status) {
      updates.is_completed = updates.status === 'done';
    }

    // Ensure the task has a valid status
    const validStatuses = getValidStatuses();
    if (!updates.status && (!todo.status || !validStatuses.includes(todo.status))) {
      // If the task doesn't have a valid status, set it based on is_completed
      updates.status = todo.is_completed ? 'done' : 'open';
    }

    todosMap.set(id, {
      ...todo,
      ...updates,
      updated_at: new Date().toISOString()
    });

    if (isOnline) {
      syncWithSupabase();
    }
  }
}

// Function to delete a todo locally
export function deleteTodo(id) {
  todosMap.delete(id);

  if (isOnline) {
    syncWithSupabase();
  }
}

// Function to get all todos
export function getAllTodos() {
  if (!isBrowser || !todosMap) {
    return [];
  }

  try {
    const validStatuses = getValidStatuses();

    return Array.from(todosMap.entries()).map(([id, value]) => {
      // Ensure each task has a valid status
      if (!value.status || !validStatuses.includes(value.status)) {
        // If the task doesn't have a valid status, set it based on is_completed
        value.status = value.is_completed ? 'done' : 'open';

        // Update the task in the map
        todosMap.set(id, {
          ...value,
          updated_at: new Date().toISOString()
        });
      }

      return {
        id,
        ...value
      };
    });
  } catch (error) {
    console.error('Error getting todos:', error);
    return [];
  }
}

// Listen for changes to the todos map
todosMap.observe(() => {
  // Trigger sync when todos change
  if (isOnline) {
    syncWithSupabase();
  }
});
