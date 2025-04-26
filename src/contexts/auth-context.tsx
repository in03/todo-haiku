import {
  createContextId,
  useContextProvider,
  useContext,
  component$,
  Slot,
  useStore,
  useVisibleTask$,
  useSignal
} from '@builder.io/qwik';
import { useNavigate } from '@builder.io/qwik-city';
import { createBrowserSupabaseClient } from '~/services/supabase';

// Define the auth state type
export interface AuthState {
  user: any | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isDevelopmentMode: boolean;
}

// Create the context
export const AuthContext = createContextId<AuthState>('auth-context');

// Hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Auth provider component
export const AuthProvider = component$(() => {
  const navigate = useNavigate();
  const supabase = useSignal<any>(null);

  // Create a store for auth state
  const authState = useStore<AuthState>({
    user: null,
    isLoading: true,
    isAuthenticated: false,
    // Set this to true during development to bypass auth
    isDevelopmentMode: true
  });

  // Initialize auth state
  useVisibleTask$(async () => {
    try {
      // Initialize Supabase client
      supabase.value = createBrowserSupabaseClient();

      if (!supabase.value) {
        console.error('Failed to create Supabase client');
        authState.isLoading = false;
        return;
      }

      // Check if we're in development mode
      if (authState.isDevelopmentMode) {
        console.log('🔑 Development mode: Using mock user');
        authState.user = {
          id: 'dev-user-id',
          email: 'dev@example.com',
          role: 'authenticated'
        };
        authState.isAuthenticated = true;
        authState.isLoading = false;
        return;
      }

      // Get current session
      const { data, error } = await supabase.value.auth.getSession();

      if (error) {
        console.error('Error getting auth session:', error);
        authState.isLoading = false;
        return;
      }

      // Update auth state
      if (data.session) {
        authState.user = data.session.user;
        authState.isAuthenticated = true;
      }

      // Set up auth state change listener
      const { data: authListener } = supabase.value.auth.onAuthStateChange(
        (event: string, session: any) => {
          console.log('Auth state changed:', event, session?.user?.email);
          authState.user = session?.user || null;
          authState.isAuthenticated = !!session;

          // Redirect based on auth state
          if (event === 'SIGNED_IN') {
            console.log('User signed in, redirecting to home');
            navigate('/');
          } else if (event === 'SIGNED_OUT') {
            console.log('User signed out, redirecting to auth');
            navigate('/auth');
          } else if (event === 'USER_UPDATED') {
            console.log('User updated');
          } else if (event === 'TOKEN_REFRESHED') {
            console.log('Token refreshed');
          }
        }
      );

      // Clean up auth listener on component unmount
      return () => {
        authListener?.subscription.unsubscribe();
      };
    } catch (error) {
      console.error('Error initializing auth:', error);
    } finally {
      authState.isLoading = false;
    }
  });

  // Provide the auth context
  useContextProvider(AuthContext, authState);

  return <Slot />;
});
