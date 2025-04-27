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
            // Use a longer delay for initial sync to ensure everything is loaded
            debouncedSync(5000);
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
    console.log('Network connection restored, scheduling sync');
    // Use a delay to ensure network is stable
    debouncedSync(3000);
  });

  window.addEventListener('offline', () => {
    isOnline = false;
  });
}

// Function to sync local changes with Supabase
export async function syncWithSupabase() {
  // Skip sync if offline or already syncing
  if (!isOnline || isSyncing) {
    console.log('Skipping Supabase sync - offline or already syncing');
    return;
  }

  // Skip sync if Supabase client is not initialized
  if (!supabase) {
    console.log('Skipping Supabase sync - Supabase client not initialized');
    return;
  }

  // Check if auth is enabled in the app
  const authEnabled = true; // Set to false during development if needed

  try {
    isSyncing = true;

    // Get current user
    const { data: sessionData, error: sessionError } = await supabase.auth.getSession();

    // Handle session error
    if (sessionError) {
      console.log('Skipping Supabase sync - session error:', sessionError.message);
      isSyncing = false;
      return;
    }

    const userId = sessionData.session?.user.id;

    // Skip sync if no authenticated user and auth is enabled
    if (!userId && authEnabled) {
      console.log('Skipping Supabase sync - no authenticated user');
      isSyncing = false;
      return;
    }

    // If auth is disabled for development, use a mock user ID
    const effectiveUserId = userId || 'dev-user-id';

    // Get all todos from Supabase
    const { data: remoteTodos, error } = await supabase
      .from('todos')
      .select('*')
      .eq('user_id', effectiveUserId);

    if (error) {
      console.error('Error fetching remote todos:', error);
      isSyncing = false;
      return;
    }

    // Log sync status
    console.log(`Syncing ${remoteTodos?.length || 0} remote todos with local data`);

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
              user_id: effectiveUserId,
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
    // Log completion
    console.log('Supabase sync completed');
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

    // Sync with Supabase if online (debounced)
    if (isOnline) {
      debouncedSync();
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
      debouncedSync();
    }
  }
}

// Function to delete a todo locally
export function deleteTodo(id) {
  todosMap.delete(id);

  if (isOnline) {
    debouncedSync();
  }
}

// Function to get all todos
export function getAllTodos() {
  if (!isBrowser || !todosMap) {
    console.log('getAllTodos: Not in browser or todosMap not available');
    return [];
  }

  try {
    const validStatuses = getValidStatuses();
    const entries = Array.from(todosMap.entries());
    console.log('getAllTodos: Found', entries.length, 'entries in todosMap');

    // If no entries in Y.js, check localStorage as fallback
    if (entries.length === 0 && isBrowser && window.localStorage) {
      try {
        const localTodos = JSON.parse(localStorage.getItem('todos') || '[]');
        console.log('getAllTodos: Found', localTodos.length, 'todos in localStorage');

        if (localTodos.length > 0) {
          // Add todos from localStorage to Y.js
          localTodos.forEach(todo => {
            if (todo.id) {
              todosMap.set(todo.id, todo);
            }
          });

          // Try again with the updated todosMap
          return getAllTodos();
        }
      } catch (e) {
        console.error('Error reading from localStorage:', e);
      }
    }

    return entries.map(([id, value]) => {
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

// Debounce function to limit sync frequency
let syncDebounceTimeout: any = null;
const debouncedSync = (delay = 2000) => {
  if (syncDebounceTimeout) {
    clearTimeout(syncDebounceTimeout);
  }

  syncDebounceTimeout = setTimeout(() => {
    if (isOnline) {
      syncWithSupabase();
    }
    syncDebounceTimeout = null;
  }, delay);
};

// Listen for changes to the todos map
todosMap.observe(() => {
  // Trigger debounced sync when todos change
  debouncedSync();
});
