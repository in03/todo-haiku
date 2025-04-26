import { component$ } from '@builder.io/qwik';
import { DocumentHead } from '@builder.io/qwik-city';
import AppLayout from '../../components/Layout';
import AuthForm from '../../components/AuthForm';

export default component$(() => {
  return (
    <AppLayout>
      <div class="py-12">
        <AuthForm />
      </div>
    </AppLayout>
  );
});

export const head: DocumentHead = {
  title: 'Authentication - Todo Haiku',
  meta: [
    {
      name: 'description',
      content: 'Sign in or create an account for Todo Haiku',
    },
  ],
};
