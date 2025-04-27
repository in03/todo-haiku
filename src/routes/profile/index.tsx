import { component$, useSignal, useVisibleTask$, $, noSerialize, useStore } from '@builder.io/qwik';
import { DocumentHead, routeLoader$ } from '@builder.io/qwik-city';
import Layout from '../../components/Layout';
import { createBrowserSupabaseClient, getServerSupabaseClient } from '../../services/supabase';
import { syncWithSupabase, getAllTodos, deleteTodo } from '../../services/yjs-sync';
import {
  isPushNotificationSupported,
  requestNotificationPermission,
  subscribeToPushNotifications
} from '../../services/push-notifications';
import { useAuth, useToggleDevMode } from '~/contexts/auth-context';
import { validateHaiku } from '~/utils/haiku-validator';
import { Card, CardHeader, CardTitle, CardContent, CardFooter, Button, Textarea } from '~/components/ui';

// Server-side route loader to get user data
export const useUserLoader = routeLoader$(async () => {
  const supabase = await getServerSupabaseClient();
  const { data } = await supabase.auth.getUser();
  return data.user;
});

export default component$(() => {
  const userLoader = useUserLoader();
  const notificationsEnabled = useSignal(false);
  const syncStatus = useSignal('');
  const supabase = useSignal<any>(null);

  // Delete all tasks modal state
  const deleteModalOpen = useSignal(false);
  const deleteConfirmationHaiku = useSignal('');
  const deleteStatus = useSignal('');
  const taskCount = useSignal(0);

  // Sample tasks state
  const showSampleTasksBanner = useSignal(false);
  const deleteSampleTasksModalOpen = useSignal(false);

  // Haiku validation state
  const validation = useStore({
    isValid: false,
    syllableCounts: [0, 0, 0],
    feedback: ''
  });

  // Get auth state from context
  const auth = useAuth();
  const toggleDevMode = useToggleDevMode();

  // Initialize Supabase client and check notification status
  useVisibleTask$(async () => {
    // Initialize Supabase client
    const client = createBrowserSupabaseClient();

    if (client) {
      // Use noSerialize to prevent serialization errors
      supabase.value = noSerialize(client);

      // No need to set up auth state change listener here
      // We're using the auth context instead
    }

    // Check notification permission
    if (isPushNotificationSupported()) {
      notificationsEnabled.value = Notification.permission === 'granted';
    }

    // Get task count
    const tasks = getAllTodos();
    taskCount.value = tasks.length;

    // Check if sample tasks banner should be shown
    const hasShownSampleTasks = localStorage.getItem('hasShownSampleTasks') === 'true';
    const hideSampleTasksBanner = localStorage.getItem('hideSampleTasksBanner') === 'true';

    // Show banner if sample tasks have been shown but not hidden by user
    showSampleTasksBanner.value = hasShownSampleTasks && !hideSampleTasksBanner && tasks.length === 0;
  });

  // Validate haiku as user types
  const validateConfirmationHaiku = $(() => {
    const result = validateHaiku(deleteConfirmationHaiku.value);
    validation.isValid = result.isValid;
    validation.syllableCounts = result.syllableCounts;
    validation.feedback = result.feedback;
  });

  // Open delete all tasks modal
  const openDeleteModal = $(() => {
    // Reset state
    deleteConfirmationHaiku.value = '';
    deleteStatus.value = '';
    validation.isValid = false;
    validation.syllableCounts = [0, 0, 0];
    validation.feedback = '';

    // Update task count
    const tasks = getAllTodos();
    taskCount.value = tasks.length;

    // Open modal
    deleteModalOpen.value = true;
  });

  // Close delete all tasks modal
  const closeDeleteModal = $(() => {
    deleteModalOpen.value = false;
  });

  // Hide sample tasks banner
  const hideSampleTasksBanner = $(() => {
    showSampleTasksBanner.value = false;
    localStorage.setItem('hideSampleTasksBanner', 'true');
  });

  // Open delete sample tasks confirmation modal
  const openDeleteSampleTasksModal = $(() => {
    deleteSampleTasksModalOpen.value = true;
  });

  // Close delete sample tasks confirmation modal
  const closeDeleteSampleTasksModal = $(() => {
    deleteSampleTasksModalOpen.value = false;
  });

  // Delete sample tasks flag
  const deleteSampleTasksFlag = $(() => {
    // Remove the sample tasks flag
    localStorage.removeItem('hasShownSampleTasks');

    // Hide the banner
    showSampleTasksBanner.value = false;

    // Close the modal
    deleteSampleTasksModalOpen.value = false;

    // Show confirmation
    syncStatus.value = 'Sample tasks reset. Refresh the page to see sample tasks.';
    setTimeout(() => {
      syncStatus.value = '';
    }, 3000);
  });

  // Delete all tasks
  const deleteAllTasks = $(async () => {
    if (!validation.isValid) {
      deleteStatus.value = 'Please enter a valid haiku to confirm deletion.';
      return;
    }

    try {
      deleteStatus.value = 'Deleting all tasks...';

      // Get all tasks
      const tasks = getAllTodos();

      // Delete each task
      for (const task of tasks) {
        deleteTodo(task.id);
      }

      // Reset sample tasks flag
      localStorage.removeItem('hasShownSampleTasks');

      // Reset hide banner flag
      localStorage.removeItem('hideSampleTasksBanner');

      // Update status
      deleteStatus.value = 'All tasks deleted successfully. Refresh the page to see sample tasks.';

      // Update task count
      taskCount.value = 0;

      // Close modal after a delay
      setTimeout(() => {
        closeDeleteModal();
      }, 3000);
    } catch (error) {
      console.error('Error deleting tasks:', error);
      deleteStatus.value = 'Error deleting tasks. Please try again.';
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

          {/* Sample Tasks Banner */}
          {showSampleTasksBanner.value && (
            <div class="mb-6 bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 px-4 py-3 rounded-md flex justify-between items-center">
              <div class="flex items-center">
                <span class="text-lg mr-2">🍃</span>
                <span>Try adding some tasks! Sample tasks will appear when you refresh the page.</span>
              </div>
              <div class="flex gap-2">
                <button
                  onClick$={hideSampleTasksBanner}
                  class="text-amber-800 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800/30 p-1 rounded-md"
                  aria-label="Close banner"
                >
                  ✕
                </button>
                <button
                  onClick$={openDeleteSampleTasksModal}
                  class="text-amber-800 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800/30 p-1 rounded-md text-xs px-2"
                >
                  Reset
                </button>
              </div>
            </div>
          )}

          {/* Delete Sample Tasks Confirmation Modal */}
          {deleteSampleTasksModalOpen.value && (
            <div class="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <Card class="w-full max-w-md">
                <CardHeader>
                  <CardTitle>Reset Sample Tasks</CardTitle>
                </CardHeader>
                <CardContent>
                  <p class="mb-4">
                    Do you want to reset the sample tasks flag? This will allow sample tasks to appear again when you refresh the page.
                  </p>
                </CardContent>
                <CardFooter class="flex justify-between border-t border-border pt-4">
                  <Button
                    variant="outline"
                    onClick$={closeDeleteSampleTasksModal}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick$={deleteSampleTasksFlag}
                  >
                    Reset Sample Tasks
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}

          {auth.isAuthenticated && auth.user ? (
            <div>
              <div class="bg-card p-6 rounded-lg mb-6">
                <h2 class="text-xl font-semibold mb-4">Account Information</h2>
                <p class="mb-2">
                  <span class="text-muted-foreground">Email:</span> {auth.user.email}
                </p>
                <p>
                  <span class="text-muted-foreground">User ID:</span> {auth.user.id}
                </p>
                {auth.isDevelopmentMode && (
                  <p class="mt-4 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 px-3 py-1 rounded-md inline-block">
                    Development Mode Active
                  </p>
                )}
              </div>

              <div class="bg-card p-6 rounded-lg mb-6">
                <h2 class="text-xl font-semibold mb-4">Sync Settings</h2>
                <p class="mb-4 text-sm text-muted-foreground">
                  Sync your local todos with the cloud to access them across devices.
                </p>
                <div class="flex flex-wrap gap-4">
                  <button
                    onClick$={syncTodos}
                    class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
                  >
                    Sync Now
                  </button>

                  <button
                    onClick$={openDeleteModal}
                    class="px-4 py-2 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90"
                    disabled={taskCount.value === 0}
                  >
                    Delete All Tasks ({taskCount.value})
                  </button>
                </div>
                {syncStatus.value && (
                  <p class="mt-2 text-sm">
                    {syncStatus.value}
                  </p>
                )}

                {/* Delete All Tasks Modal */}
                {deleteModalOpen.value && (
                  <div class="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                    <Card class="w-full max-w-md">
                      <CardHeader>
                        <CardTitle class="text-destructive">Delete All Tasks</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p class="mb-4">
                          You are about to delete all {taskCount.value} tasks. This action cannot be undone.
                        </p>
                        <p class="mb-4 font-medium">
                          To confirm, please write a deletion themed haiku:
                        </p>

                        <Textarea
                          value={deleteConfirmationHaiku.value}
                          onInput$={(e: any) => {
                            deleteConfirmationHaiku.value = e.target.value;
                            validateConfirmationHaiku();
                          }}
                          class="font-serif h-32 mb-4"
                          placeholder="Write a deletion themed haiku (5-7-5 syllables)"
                        />

                        <div class="bg-muted/20 p-3 rounded-md">
                          <div class="flex justify-between">
                            <div class={`text-xs px-2 py-1 rounded-md ${validation.syllableCounts[0] === 5 ? 'bg-green-500/20 text-green-500' : 'text-muted-foreground'}`}>
                              Line 1: {validation.syllableCounts[0] || 0}/5 syllables
                            </div>
                            <div class={`text-xs px-2 py-1 rounded-md ${validation.syllableCounts[1] === 7 ? 'bg-green-500/20 text-green-500' : 'text-muted-foreground'}`}>
                              Line 2: {validation.syllableCounts[1] || 0}/7 syllables
                            </div>
                            <div class={`text-xs px-2 py-1 rounded-md ${validation.syllableCounts[2] === 5 ? 'bg-green-500/20 text-green-500' : 'text-muted-foreground'}`}>
                              Line 3: {validation.syllableCounts[2] || 0}/5 syllables
                            </div>
                          </div>
                          <p class={`text-sm mt-1 ${validation.isValid ? 'text-green-500' : 'text-amber-500'}`}>
                            {validation.feedback}
                          </p>
                        </div>

                        {deleteStatus.value && (
                          <p class="mt-4 text-sm">
                            {deleteStatus.value}
                          </p>
                        )}
                      </CardContent>
                      <CardFooter class="flex justify-between border-t border-border pt-4">
                        <Button
                          variant="outline"
                          onClick$={closeDeleteModal}
                        >
                          Cancel
                        </Button>
                        <Button
                          variant="destructive"
                          onClick$={deleteAllTasks}
                          disabled={!validation.isValid}
                        >
                          Delete All Tasks
                        </Button>
                      </CardFooter>
                    </Card>
                  </div>
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
              {auth.isDevelopmentMode ? (
                <div>
                  <p class="mb-4 text-sm">
                    You are in development mode but not authenticated.
                  </p>
                  <button
                    onClick$={toggleDevMode}
                    class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
                  >
                    Enable Dev User
                  </button>
                </div>
              ) : (
                <a
                  href="/auth"
                  class="px-4 py-2 bg-accent text-accent-foreground rounded-md hover:bg-accent/90"
                >
                  Sign In
                </a>
              )}
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
