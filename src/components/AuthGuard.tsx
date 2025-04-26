import { component$, Slot } from '@builder.io/qwik';
import { useNavigate } from '@builder.io/qwik-city';
import { useAuth } from '~/contexts/auth-context';

export const AuthGuard = component$(() => {
  const auth = useAuth();
  const navigate = useNavigate();

  // If auth is still loading, show a loading indicator
  if (auth.isLoading) {
    return (
      <div class="flex justify-center items-center h-screen">
        <div class="text-lg text-muted-foreground">Loading...</div>
      </div>
    );
  }

  // If in development mode or authenticated, render the children
  if (auth.isDevelopmentMode || auth.isAuthenticated) {
    return <Slot />;
  }

  // If not authenticated, redirect to login
  navigate('/auth');
  
  // Show a message while redirecting
  return (
    <div class="flex justify-center items-center h-screen">
      <div class="text-lg text-muted-foreground">Redirecting to login...</div>
    </div>
  );
});
