import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { DocumentHead, routeLoader$ } from '@builder.io/qwik-city';
import Layout from '../../components/Layout';
import { createBrowserSupabaseClient, getServerSupabaseClient } from '../../services/supabase';
import { syncWithSupabase } from '../../services/yjs-sync';
import {
  isPushNotificationSupported,
  requestNotificationPermission,
  subscribeToPushNotifications
} from '../../services/push-notifications';

// Server-side route loader to get user data
export const useUserLoader = routeLoader$(async () => {
  const supabase = await getServerSupabaseClient();
  const { data } = await supabase.auth.getUser();
  return data.user;
});

export default component$(() => {
  const userLoader = useUserLoader();
  const user = useSignal<any>(userLoader.value);
  const notificationsEnabled = useSignal(false);
  const syncStatus = useSignal('');
  const supabase = useSignal<any>(null);

  // Initialize Supabase client and check notification status
  useVisibleTask$(async () => {
    // Initialize Supabase client
    supabase.value = createBrowserSupabaseClient();

    // Set up auth state change listener
    supabase.value.auth.onAuthStateChange((event, session) => {
      user.value = session?.user || null;
    });

    // Check notification permission
    if (isPushNotificationSupported()) {
      notificationsEnabled.value = Notification.permission === 'granted';
    }
  });

  // Enable push notifications
  const enableNotifications = $(async () => {
    const permissionGranted = await requestNotificationPermission();
    if (permissionGranted) {
      const subscription = await subscribeToPushNotifications();
      notificationsEnabled.value = !!subscription;
    }
  });

  // Sync todos with Supabase
  const syncTodos = $(async () => {
    syncStatus.value = 'Syncing...';
    try {
      await syncWithSupabase();
      syncStatus.value = 'Sync completed successfully!';
      setTimeout(() => {
        syncStatus.value = '';
      }, 3000);
    } catch (error) {
      syncStatus.value = 'Sync failed. Please try again.';
    }
  });

  return (
    <Layout>
      <div class="py-8">
        <div class="zen-container">
          <h1 class="text-3xl font-bold mb-6">Your Profile</h1>

          {user.value ? (
            <div>
              <div class="bg-card p-6 rounded-lg mb-6">
                <h2 class="text-xl font-semibold mb-4">Account Information</h2>
                <p class="mb-2">
                  <span class="text-muted-foreground">Email:</span> {user.value.email}
                </p>
                <p>
                  <span class="text-muted-foreground">User ID:</span> {user.value.id}
                </p>
              </div>

              <div class="bg-card p-6 rounded-lg mb-6">
                <h2 class="text-xl font-semibold mb-4">Sync Settings</h2>
                <p class="mb-4 text-sm text-muted-foreground">
                  Sync your local todos with the cloud to access them across devices.
                </p>
                <button
                  onClick$={syncTodos}
                  class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
                >
                  Sync Now
                </button>
                {syncStatus.value && (
                  <p class="mt-2 text-sm">
                    {syncStatus.value}
                  </p>
                )}
              </div>

              <div class="bg-card p-6 rounded-lg">
                <h2 class="text-xl font-semibold mb-4">Notification Settings</h2>
                {isPushNotificationSupported() ? (
                  <div>
                    <p class="mb-4 text-sm text-muted-foreground">
                      {notificationsEnabled.value
                        ? 'Push notifications are enabled.'
                        : 'Enable push notifications to receive reminders for your tasks.'}
                    </p>
                    {!notificationsEnabled.value && (
                      <button
                        onClick$={enableNotifications}
                        class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
                      >
                        Enable Notifications
                      </button>
                    )}
                  </div>
                ) : (
                  <p class="text-sm text-muted-foreground">
                    Your browser does not support push notifications.
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div class="text-center py-12">
              <p class="text-muted-foreground mb-4">
                Please sign in to view your profile.
              </p>
              <a
                href="/auth"
                class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
              >
                Sign In
              </a>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
});

export const head: DocumentHead = {
  title: 'Profile - Todo Haiku',
  meta: [
    {
      name: 'description',
      content: 'Manage your Todo Haiku profile and settings',
    },
  ],
};
