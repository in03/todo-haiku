# Todo Haiku

A mindful todo app where tasks are written as haikus, built with Qwik, Supabase, and Y.js.

## Features

- **Haiku Validation**: Tasks must be written as haikus (5-7-5 syllable pattern)
- **Syllable Counting**: Real-time syllable counting for each line using ML via ONNX
- **Feedback**: Serverless functions for LLM quality rating, critique, and mood.
- **Kanban Board**: Organize tasks in a beautiful kanban board with haiku-themed statuses
- **Local-First Architecture**: Works offline with Y.js and IndexedDB
- **Cloud Sync**: Syncs with Supabase when online
- **Responsive Design**: Works on desktop and mobile with optimized layouts
- **Dark Mode**: Mindful, minimal, Japanese-inspired design

### Coming Soon
- **PWA Support**: Install as a Progressive Web App
- **Push Notifications**: Get reminders for your tasks
- **Authentication**: Secure user accounts with Supabase Auth

## Tech Stack

- **Frontend**: Qwik, TypeScript, TailwindCSS, ShadCN UI
- **Backend**: Supabase (PostgreSQL, Auth)
- **Local-First**: Y.js, IndexedDB
- **Drag and Drop**: SortableJS for kanban functionality
- **Syllable Counting**: ONNXRuntime for local ML haiku validation using [Kaggle dataset](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary?resource=download)
- **Feedback**: Supabase Serverless Functions for NLP haiku feedback
- **PWA**: Service Workers, Web Push API (coming soon)

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   bun install
   ```
3. Set up Supabase (this will install and start Supabase locally):
   ```
   bun run setup
   ```
4. Run the development server:
   ```
   bun run dev
   ```

### Authentication (Coming Soon)

Authentication will be implemented using Supabase Auth with protected routes:

```typescript
// Example of future implementation
export const useAuthCheck = routeLoader$(async (requestEv) => {
  const supabase = await getServerSupabaseClient();
  const { data } = await supabase.auth.getSession();

  if (!data.session) {
    throw requestEv.redirect(302, '/auth');
  }

  return data.session.user;
});
```

## Environment Variables

Copy the `.env.example` file to `.env` and update the values:

```bash
cp .env.example .env
```

Then edit the `.env` file with your Supabase credentials:

```
# Supabase configuration
VITE_SUPABASE_URL=your-supabase-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key

# Push notifications (for production)
VITE_VAPID_PUBLIC_KEY=your-vapid-public-key
```

## License

MIT
