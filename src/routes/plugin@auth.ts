import { routeLoader$ } from '@builder.io/qwik-city';
import { getServerSupabaseClient } from '../services/supabase';

// This is a route loader that will check if the user is authenticated
// It will run on all routes that include this plugin
export const useAuthCheck = routeLoader$(async (requestEv) => {
  // For development, just return a mock user to avoid redirect loops
  return {
    id: 'mock-user-id',
    email: 'user@example.com',
    role: 'authenticated'
  };

  /* Uncomment this when you have a real Supabase setup
  // Prevent redirect loops by checking if we're already on the auth page
  const url = new URL(requestEv.request.url);
  if (url.pathname === '/auth') {
    return null;
  }

  try {
    const supabase = await getServerSupabaseClient();

    // Handle case where Supabase client is null
    if (!supabase) {
      console.error('Supabase client not initialized');
      return null;
    }

    const { data } = await supabase.auth.getSession();

    // If no session, redirect to login
    if (!data.session) {
      throw requestEv.redirect(302, '/auth');
    }

    // Return the user for use in the route
    return data.session.user;
  } catch (error) {
    // Only redirect if it's not already a redirect error
    if (!(error instanceof Response)) {
      console.error('Auth check error:', error);
    }
    throw error;
  }
  */
});
