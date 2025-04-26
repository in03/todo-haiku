import { createClient } from '@supabase/supabase-js';
import type { Database } from '../types/supabase';
import { server$ } from '@builder.io/qwik-city';

// These would be environment variables in a real app
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://example.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNvbWVyZWZlcmVuY2UiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTYxMzA5ODU0MCwiZXhwIjoxOTI4Njc0NTQwfQ.examplekey';

// Validate URL format
const isValidUrl = (url: string) => {
  try {
    new URL(url);
    return true;
  } catch (e) {
    console.error('Invalid Supabase URL:', url);
    return false;
  }
};

// Create a standard Supabase client for direct API access
export const supabase = isValidUrl(supabaseUrl)
  ? createClient<Database>(supabaseUrl, supabaseAnonKey)
  : null;

// Create a browser client
export const createBrowserSupabaseClient = () => {
  if (!isValidUrl(supabaseUrl)) {
    console.error('Invalid Supabase URL. Check your environment variables.');
    return null;
  }

  // Create client with auth persistence
  return createClient<Database>(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: true,
      storageKey: 'todo-haiku-auth',
      autoRefreshToken: true,
      detectSessionInUrl: true
    }
  });
};

// Create a server client using server$
export const getServerSupabaseClient = server$(() => {
  if (!isValidUrl(supabaseUrl)) {
    console.error('Invalid Supabase URL. Check your environment variables.');
    return null;
  }
  return createClient<Database>(
    supabaseUrl,
    supabaseAnonKey
  );
});

// Auth helper functions that use the standard client
// These are provided for compatibility with existing code
export async function signUp(email: string, password: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.signUp({
    email,
    password,
  });
  return { data, error };
}

export async function signIn(email: string, password: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  return { data, error };
}

export async function signOut() {
  if (!supabase) {
    return { error: new Error('Supabase client not initialized') };
  }

  const { error } = await supabase.auth.signOut();
  return { error };
}

export async function getSession() {
  if (!supabase) {
    return { data: { session: null }, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.getSession();
  return { data, error };
}

// OAuth sign-in functions
export async function signInWithGoogle(redirectTo?: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: redirectTo || window.location.origin + '/auth/callback'
    }
  });

  return { data, error };
}

export async function signInWithGithub(redirectTo?: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: redirectTo || window.location.origin + '/auth/callback'
    }
  });

  return { data, error };
}

// Magic link authentication
export async function signInWithMagicLink(email: string, redirectTo?: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo: redirectTo || window.location.origin + '/auth/callback'
    }
  });

  return { data, error };
}

export async function getTodos(userId: string) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase
    .from('todos')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  return { data, error };
}

export async function createTodo(todo: {
  title: string;
  content: string;
  user_id: string;
  is_completed: boolean;
}) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase
    .from('todos')
    .insert([todo])
    .select();

  return { data, error };
}

export async function updateTodo(id: string, updates: {
  title?: string;
  content?: string;
  is_completed?: boolean;
}) {
  if (!supabase) {
    return { data: null, error: new Error('Supabase client not initialized') };
  }

  const { data, error } = await supabase
    .from('todos')
    .update(updates)
    .eq('id', id)
    .select();

  return { data, error };
}

export async function deleteTodo(id: string) {
  if (!supabase) {
    return { error: new Error('Supabase client not initialized') };
  }

  const { error } = await supabase
    .from('todos')
    .delete()
    .eq('id', id);

  return { error };
}
