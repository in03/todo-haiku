import { component$, Slot, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { Link, useNavigate } from '@builder.io/qwik-city';
import { createBrowserSupabaseClient } from '../services/supabase';
import { registerServiceWorker, requestNotificationPermission } from '../services/push-notifications';

export default component$(() => {
  const user = useSignal<any>(null);
  const isMenuOpen = useSignal(false);
  const supabase = useSignal<any>(null);
  const navigate = useNavigate();

  // Initialize Supabase client and check if user is logged in
  useVisibleTask$(async ({ track }) => {
    try {
      // Initialize Supabase client
      const client = createBrowserSupabaseClient();

      if (!client) {
        console.error('Failed to create Supabase client');
        return;
      }

      supabase.value = client;

      // Get session
      const { data } = await client.auth.getSession();
      user.value = data.session?.user || null;

      // Set up auth state change listener
      const { data: authListener } = client.auth.onAuthStateChange((event, session) => {
        user.value = session?.user || null;
      });

      // Register service worker and request notification permission
      try {
        const registration = await registerServiceWorker();
        if (registration) {
          await requestNotificationPermission();
        }
      } catch (error) {
        console.error('Error registering service worker:', error);
      }

      // Clean up auth listener on component unmount
      return () => {
        authListener?.subscription.unsubscribe();
      };
    } catch (error) {
      console.error('Error in Layout useVisibleTask$:', error);
    }
  });

  // Handle sign out
  const handleSignOut = $(async () => {
    if (supabase.value) {
      await supabase.value.auth.signOut();
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
                {user.value ? (
                  <div>
                    <div class="px-4 py-2 text-sm text-muted-foreground border-b border-border">
                      {user.value.email}
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
