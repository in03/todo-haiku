import { component$, Slot, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { Link, useNavigate } from '@builder.io/qwik-city';
import { createBrowserSupabaseClient } from '../services/supabase';
import { registerServiceWorker, requestNotificationPermission } from '../services/push-notifications';
import { useAuth } from '~/contexts/auth-context';

export default component$(() => {
  const isMenuOpen = useSignal(false);
  const navigate = useNavigate();
  const auth = useAuth() as {
    user: any;
    isAuthenticated: boolean;
    isDevelopmentMode: boolean;
  };

  // Register service worker
  useVisibleTask$(async () => {
    try {
      // Register service worker and request notification permission
      const registration = await registerServiceWorker();
      if (registration) {
        await requestNotificationPermission();
      }
    } catch (error) {
      console.error('Error registering service worker:', error);
    }
  });

  // Handle sign out
  const handleSignOut = $(async () => {
    try {
      // Create a new Supabase client for sign out
      const client = createBrowserSupabaseClient();
      if (client) {
        await client.auth.signOut();
        // Auth context will handle the redirect
      }
    } catch (error) {
      console.error('Error signing out:', error);
    }
  });

  // Toggle development mode
  const toggleDevMode = $(() => {
    auth.isDevelopmentMode = !auth.isDevelopmentMode;
    if (!auth.isDevelopmentMode && !auth.isAuthenticated) {
      navigate('/auth');
    }
  });

  return (
    <div class="min-h-screen bg-background text-foreground flex flex-col">
      <header class="border-b border-border">
        <div class="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link href="/" class="text-xl font-semibold">
            Todo Haiku
          </Link>

          <div class="relative">
            <button
              onClick$={() => isMenuOpen.value = !isMenuOpen.value}
              class="p-2 rounded-md hover:bg-muted"
              aria-label="Menu"
            >
              ☰
            </button>

            {isMenuOpen.value && (
              <div class="absolute right-0 mt-2 w-48 bg-card border border-border rounded-md shadow-lg z-10">
                {auth.isAuthenticated ? (
                  <div>
                    <div class="px-4 py-2 text-sm text-muted-foreground border-b border-border">
                      {auth.user?.email || 'Authenticated User'}
                    </div>
                    <Link href="/" class="block px-4 py-2 text-sm hover:bg-muted">
                      Home
                    </Link>
                    <Link href="/profile" class="block px-4 py-2 text-sm hover:bg-muted">
                      Profile
                    </Link>
                    <button
                      onClick$={handleSignOut}
                      class="block w-full text-left px-4 py-2 text-sm hover:bg-muted"
                    >
                      Sign Out
                    </button>
                  </div>
                ) : (
                  <div>
                    <Link href="/" class="block px-4 py-2 text-sm hover:bg-muted">
                      Home
                    </Link>
                    <Link href="/auth" class="block px-4 py-2 text-sm hover:bg-muted">
                      Sign In
                    </Link>
                  </div>
                )}

                {/* Development mode toggle */}
                <div class="border-t border-border mt-2 pt-2">
                  <button
                    onClick$={toggleDevMode}
                    class="block w-full text-left px-4 py-2 text-xs text-muted-foreground hover:bg-muted"
                  >
                    {auth.isDevelopmentMode ? 'Disable' : 'Enable'} Dev Mode
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main class="flex-1">
        <Slot />
      </main>

      <footer class="border-t border-border py-6">
        <div class="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Todo Haiku - A mindful task manager</p>
          <p class="mt-1">Write your tasks as haikus for a more mindful experience</p>
        </div>
      </footer>
    </div>
  );
});
