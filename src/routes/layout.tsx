import { component$, Slot } from '@builder.io/qwik';
import { AuthProvider } from '~/contexts/auth-context';

export default component$(() => {
  return (
    <AuthProvider>
      <Slot />
    </AuthProvider>
  );
});
