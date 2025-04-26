import { component$, Slot } from '@builder.io/qwik';
import { useAuthCheck } from '../plugin@auth';
import { useNavigate } from '@builder.io/qwik-city';

export default component$(() => {
  // This will redirect to /auth if not authenticated
  const auth = useAuthCheck();
  const navigate = useNavigate();

  // If auth check returned null (error case), show a message
  if (auth.value === null) {
    return (
      <div class="p-8 text-center">
        <h1 class="text-2xl font-bold mb-4">Authentication Error</h1>
        <p class="mb-4">There was an error checking your authentication status.</p>
        <button
          onClick$={() => navigate('/')}
          class="px-4 py-2 bg-accent text-accent-foreground rounded-md"
        >
          Return to Home
        </button>
      </div>
    );
  }

  return <Slot />;
});
