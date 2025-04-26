import { component$, useSignal, $, useVisibleTask$, noSerialize } from '@builder.io/qwik';
import { useNavigate } from '@builder.io/qwik-city';
import { createBrowserSupabaseClient } from '../services/supabase';
import { useAuth, useToggleDevMode } from '~/contexts/auth-context';
import { Provider } from '@supabase/supabase-js';

export default component$(() => {
  const email = useSignal('');
  const password = useSignal('');
  const isSignUp = useSignal(false);
  const useMagicLink = useSignal(false);
  const error = useSignal('');
  const success = useSignal('');
  const isLoading = useSignal(false);
  const supabase = useSignal<any>(null);
  const navigate = useNavigate();
  const auth = useAuth();
  const toggleDevMode = useToggleDevMode();

  // If already authenticated or in dev mode, redirect to home
  if (auth.isAuthenticated || auth.isDevelopmentMode) {
    navigate('/');
    return null;
  }

  // Initialize Supabase client
  useVisibleTask$(() => {
    const client = createBrowserSupabaseClient();
    if (client) {
      supabase.value = noSerialize(client);
    }
  });

  // Handle email/password authentication
  const handleSubmit = $(async () => {
    // Validate inputs
    if (useMagicLink.value) {
      if (!email.value) {
        error.value = 'Please enter your email';
        return;
      }
    } else {
      if (!email.value || !password.value) {
        error.value = 'Please enter both email and password';
        return;
      }
    }

    if (!supabase.value) {
      error.value = 'Authentication client not initialized';
      return;
    }

    error.value = '';
    success.value = '';
    isLoading.value = true;

    try {
      // Handle magic link authentication
      if (useMagicLink.value) {
        const { error: magicLinkError } = await supabase.value.auth.signInWithOtp({
          email: email.value,
          options: {
            emailRedirectTo: window.location.origin + '/auth/callback'
          }
        });

        if (magicLinkError) {
          error.value = magicLinkError.message;
        } else {
          // Show success message
          success.value = 'Check your email for a magic link to sign in';
        }
      }
      // Handle sign up
      else if (isSignUp.value) {
        const { error: signUpError } = await supabase.value.auth.signUp({
          email: email.value,
          password: password.value,
          options: {
            emailRedirectTo: window.location.origin + '/auth/callback'
          }
        });

        if (signUpError) {
          error.value = signUpError.message;
        } else {
          // Show success message
          success.value = 'Check your email for a confirmation link';
        }
      }
      // Handle password login
      else {
        const { error: signInError } = await supabase.value.auth.signInWithPassword({
          email: email.value,
          password: password.value,
        });

        if (signInError) {
          error.value = signInError.message;
        } else {
          // Redirect to home page on successful login
          navigate('/');
        }
      }
    } catch (e) {
      error.value = 'An unexpected error occurred';
      console.error(e);
    } finally {
      isLoading.value = false;
    }
  });

  // Handle OAuth sign-in
  const handleOAuthSignIn = $(async (provider: Provider) => {
    if (!supabase.value) {
      error.value = 'Authentication client not initialized';
      return;
    }

    error.value = '';
    isLoading.value = true;

    try {
      const { error: signInError } = await supabase.value.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: window.location.origin + '/auth/callback'
        }
      });

      if (signInError) {
        error.value = signInError.message;
        isLoading.value = false;
      }
      // No need to set isLoading to false on success as we're redirecting
    } catch (e) {
      error.value = 'An unexpected error occurred';
      console.error(e);
      isLoading.value = false;
    }
  });

  return (
    <div class="zen-container max-w-md mx-auto">
      <div class="bg-card p-6 rounded-lg shadow-lg">
        <h2 class="text-2xl font-semibold mb-6 text-center">
          {isSignUp.value ? 'Create an Account' : 'Sign In'}
        </h2>

        {error.value && (
          <div class="mb-4 p-3 bg-destructive/20 text-destructive rounded-md text-sm">
            {error.value}
          </div>
        )}

        {success.value && (
          <div class="mb-4 p-3 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded-md text-sm">
            {success.value}
          </div>
        )}

        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium mb-1" for="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email.value}
              onInput$={(e: any) => email.value = e.target.value}
              class="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-1 focus:ring-accent"
              placeholder="Enter your email"
            />
          </div>

          {!useMagicLink.value && (
            <div>
              <label class="block text-sm font-medium mb-1" for="password">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password.value}
                onInput$={(e: any) => password.value = e.target.value}
                class="w-full px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-1 focus:ring-accent"
                placeholder="Enter your password"
              />
            </div>
          )}

          <button
            onClick$={handleSubmit}
            disabled={isLoading.value}
            class="w-full py-2 px-4 bg-accent text-accent-foreground rounded-md hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading.value
              ? 'Loading...'
              : useMagicLink.value
                ? 'Send Magic Link'
                : isSignUp.value
                  ? 'Sign Up'
                  : 'Sign In'
            }
          </button>

          {/* OAuth Providers */}
          <div class="relative my-6">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-border"></div>
            </div>
            <div class="relative flex justify-center text-xs">
              <span class="bg-card px-2 text-muted-foreground">
                Or continue with
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <button
              onClick$={() => handleOAuthSignIn('github')}
              disabled={isLoading.value}
              class="flex items-center justify-center gap-2 rounded-md border border-border bg-background py-2 px-4 text-sm hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github">
                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"></path>
                <path d="M9 18c-4.51 2-5-2-7-2"></path>
              </svg>
              GitHub
            </button>

            <button
              onClick$={() => handleOAuthSignIn('google')}
              disabled={isLoading.value}
              class="flex items-center justify-center gap-2 rounded-md border border-border bg-background py-2 px-4 text-sm hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M17.13 17.21c-.95.46-2 .73-3.13.73-4.41 0-8-3.59-8-8 0-4.41 3.59-8 8-8 2.29 0 4.35.95 5.83 2.5"></path>
                <path d="M6.87 6.79C7.82 6.33 8.87 6.06 10 6.06c4.41 0 8 3.59 8 8 0 4.41-3.59 8-8 8-2.29 0-4.35-.95-5.83-2.5"></path>
              </svg>
              Google
            </button>
          </div>

          <div class="text-center mt-6">
            {!useMagicLink.value && (
              <button
                onClick$={() => isSignUp.value = !isSignUp.value}
                class="text-sm text-accent hover:underline"
              >
                {isSignUp.value
                  ? 'Already have an account? Sign In'
                  : 'Need an account? Sign Up'
                }
              </button>
            )}

            <div class="mt-3">
              <button
                onClick$={() => {
                  useMagicLink.value = !useMagicLink.value;
                  if (useMagicLink.value) {
                    isSignUp.value = false;
                  }
                }}
                class="text-sm text-accent hover:underline"
              >
                {useMagicLink.value
                  ? 'Use password instead'
                  : 'Use magic link (passwordless)'
                }
              </button>
            </div>

            <div class="mt-6 pt-4 border-t border-border">
              <button
                onClick$={toggleDevMode}
                class="text-xs text-muted-foreground hover:text-accent"
              >
                Enable Development Mode
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});
