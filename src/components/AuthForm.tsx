import { component$, useSignal, $, useVisibleTask$ } from '@builder.io/qwik';
import { useNavigate } from '@builder.io/qwik-city';
import { createBrowserSupabaseClient } from '../services/supabase';

export default component$(() => {
  const email = useSignal('');
  const password = useSignal('');
  const isSignUp = useSignal(false);
  const error = useSignal('');
  const isLoading = useSignal(false);
  const supabase = useSignal<any>(null);
  const navigate = useNavigate();

  // Initialize Supabase client
  useVisibleTask$(() => {
    supabase.value = createBrowserSupabaseClient();
  });

  const handleSubmit = $(async () => {
    if (!email.value || !password.value) {
      error.value = 'Please enter both email and password';
      return;
    }

    if (!supabase.value) {
      error.value = 'Authentication client not initialized';
      return;
    }

    error.value = '';
    isLoading.value = true;

    try {
      if (isSignUp.value) {
        const { error: signUpError } = await supabase.value.auth.signUp({
          email: email.value,
          password: password.value,
        });

        if (signUpError) {
          error.value = signUpError.message;
        } else {
          // Redirect to confirmation page or show message
          error.value = 'Check your email for a confirmation link';
        }
      } else {
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

          <button
            onClick$={handleSubmit}
            disabled={isLoading.value}
            class="w-full py-2 px-4 bg-accent text-accent-foreground rounded-md hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading.value
              ? 'Loading...'
              : isSignUp.value
                ? 'Sign Up'
                : 'Sign In'
            }
          </button>

          <div class="text-center mt-4">
            <button
              onClick$={() => isSignUp.value = !isSignUp.value}
              class="text-sm text-accent hover:underline"
            >
              {isSignUp.value
                ? 'Already have an account? Sign In'
                : 'Need an account? Sign Up'
              }
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});
