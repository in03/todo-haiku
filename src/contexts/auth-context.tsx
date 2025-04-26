import {
  createContextId,
  useContextProvider,
  useContext,
  component$,
  Slot,
  useStore,
  useVisibleTask$,
  useSignal,
  noSerialize,
  $
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

// Create the contexts
export const AuthContext = createContextId<AuthState>('auth-context');
export const ToggleDevModeContext = createContextId<any>('toggle-dev-mode-context');
export const SignOutContext = createContextId<any>('sign-out-context');

// Hooks to use the contexts
export const useAuth = () => useContext(AuthContext);
export const useToggleDevMode = () => useContext(ToggleDevModeContext);
export const useSignOut = () => useContext(SignOutContext);

// Auth provider component
export const AuthProvider = component$(() => {
  const navigate = useNavigate();
  // Use any type for supabase client with noSerialize
  const supabase = useSignal<any>(null);

  // Create a store for auth state
  const authState = useStore<AuthState>({
    user: null,
    isLoading: true,
    isAuthenticated: false,
    // Set this to true during development to bypass auth
    isDevelopmentMode: true
  });

  // Method to toggle development mode - using $ to make it a QRL
  const toggleDevMode = $(() => {
    console.log('Toggling dev mode from', authState.isDevelopmentMode, 'to', !authState.isDevelopmentMode);
    authState.isDevelopmentMode = !authState.isDevelopmentMode;

    if (authState.isDevelopmentMode) {
      // Enable dev mode - set mock user
      authState.user = {
        id: 'dev-user-id',
        email: 'dev@example.com',
        role: 'authenticated'
      };
      authState.isAuthenticated = true;
      navigate('/');
    } else {
      // Disable dev mode - check for real session
      if (supabase.value) {
        supabase.value.auth.getSession().then(({ data }: { data: any }) => {
          if (data.session) {
            // Real session exists
            authState.user = data.session.user;
            authState.isAuthenticated = true;
          } else {
            // No real session - sign out
            authState.user = null;
            authState.isAuthenticated = false;
            navigate('/auth');
          }
        });
      } else {
        // No supabase client - sign out
        authState.user = null;
        authState.isAuthenticated = false;
        navigate('/auth');
      }
    }
  });

  // Method to sign out - using $ to make it a QRL
  const signOut = $(async () => {
    console.log('Signing out...');
    if (authState.isDevelopmentMode) {
      console.log('Dev mode active, clearing mock user');
      authState.isDevelopmentMode = false;
      authState.user = null;
      authState.isAuthenticated = false;
      navigate('/auth');
      return;
    }

    if (supabase.value) {
      try {
        await supabase.value.auth.signOut();
        authState.user = null;
        authState.isAuthenticated = false;
        navigate('/auth');
      } catch (error) {
        console.error('Error signing out:', error);
      }
    }
  });

  // Initialize auth state
  useVisibleTask$(async () => {
    try {
      // Initialize Supabase client
      const client = createBrowserSupabaseClient();

      if (!client) {
        console.error('Failed to create Supabase client');
        authState.isLoading = false;
        return;
      }

      // Use noSerialize to prevent serialization errors
      supabase.value = noSerialize(client);

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

  // Provide the contexts
  useContextProvider(AuthContext, authState);
  useContextProvider(ToggleDevModeContext, toggleDevMode);
  useContextProvider(SignOutContext, signOut);

  return <Slot />;
});
