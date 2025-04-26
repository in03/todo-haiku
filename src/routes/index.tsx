import { component$ } from '@builder.io/qwik';
import { DocumentHead } from '@builder.io/qwik-city';
import AppLayout from '../components/Layout';
import { ResponsiveKanban } from '../components/kanban/ResponsiveKanban';

export default component$(() => {
  return (
    <AppLayout>
      <div class="py-8">
        <div class="zen-container text-center mb-12">
          <h1 class="text-4xl font-bold mb-4">Todo Haiku 🍃</h1>
          <p class="text-xl text-muted-foreground">
            A mindful task manager where every task is a haiku
          </p>
        </div>

        <div class="zen-divider mx-auto" />

        <ResponsiveKanban />
      </div>
    </AppLayout>
  );
});

export const head: DocumentHead = {
  title: 'Todo Haiku - A mindful Task Manager',
  meta: [
    {
      name: 'description',
      content: 'A minimal, poetic todo app where tasks are written as haikus',
    },
    {
      name: 'theme-color',
      content: '#000000',
    },
  ],
  links: [
    {
      rel: 'manifest',
      href: '/manifest.json',
    },
    {
      rel: 'icon',
      href: '/icons/icon-192x192.png',
    },
    {
      rel: 'apple-touch-icon',
      href: '/icons/icon-192x192.png',
    },
    {
      rel: 'stylesheet',
      href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+JP&display=swap',
    },
  ],
};
