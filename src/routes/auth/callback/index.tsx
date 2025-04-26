import { component$, noSerialize } from '@builder.io/qwik';
import { useNavigate } from '@builder.io/qwik-city';
import { useVisibleTask$ } from '@builder.io/qwik';
import { createBrowserSupabaseClient } from '~/services/supabase';

export default component$(() => {
  const navigate = useNavigate();

  // Handle the OAuth callback
  useVisibleTask$(async () => {
    try {
      const client = createBrowserSupabaseClient();

      if (!client) {
        console.error('Failed to create Supabase client');
        navigate('/auth?error=client_init_failed');
        return;
      }

      // Use noSerialize to prevent serialization errors
      const supabase = noSerialize(client);

      // Get the URL hash
      const hash = window.location.hash;

      // Process the callback
      const { error } = await supabase.auth.getSession();

      if (error) {
        console.error('Error getting session:', error);
        navigate('/auth?error=session_error');
        return;
      }

      // Redirect to the home page
      navigate('/');
    } catch (error) {
      console.error('Error in auth callback:', error);
      navigate('/auth?error=unknown');
    }
  });

  return (
    <div class="flex justify-center items-center h-screen">
      <div class="text-center">
        <h2 class="text-2xl font-semibold mb-4">Signing you in...</h2>
        <div class="animate-pulse text-muted-foreground">Please wait while we complete the authentication process.</div>
      </div>
    </div>
  );
});
