// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// Check if the browser supports service workers and push notifications
export function isPushNotificationSupported() {
  return isBrowser && 'serviceWorker' in navigator && 'PushManager' in window;
}

// Register the service worker
export async function registerServiceWorker() {
  if (!isPushNotificationSupported()) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js');
    return registration;
  } catch (error) {
    console.error('Service worker registration failed:', error);
    return false;
  }
}

// Request permission for push notifications
export async function requestNotificationPermission() {
  if (!isPushNotificationSupported()) {
    return false;
  }

  try {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  } catch (error) {
    console.error('Error requesting notification permission:', error);
    return false;
  }
}

// Subscribe to push notifications
export async function subscribeToPushNotifications() {
  if (!isPushNotificationSupported()) {
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;

    // This would be your VAPID public key from your server
    const vapidPublicKey = 'YOUR_VAPID_PUBLIC_KEY';

    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey)
    });

    return subscription;
  } catch (error) {
    console.error('Error subscribing to push notifications:', error);
    return null;
  }
}

// Send a push notification (this would typically be done from your server)
export async function sendPushNotification(title, body, url = '/') {
  // In a real app, you would send this data to your server
  // which would then use the Web Push API to send the notification
  console.log('Would send push notification:', { title, body, url });

  // For demo purposes, we'll just show a local notification
  if (Notification.permission === 'granted') {
    const registration = await navigator.serviceWorker.ready;
    registration.showNotification(title, {
      body,
      icon: '/icons/icon-192x192.png',
      badge: '/icons/icon-72x72.png',
      data: { url }
    });
  }
}

// Schedule a reminder notification for a todo
export async function scheduleReminderNotification(todo, reminderTime) {
  const timeUntilReminder = new Date(reminderTime).getTime() - Date.now();

  if (timeUntilReminder <= 0) {
    return;
  }

  setTimeout(() => {
    sendPushNotification(
      'Todo Reminder',
      `Reminder for your task: ${todo.title}`,
      `/todo/${todo.id}`
    );
  }, timeUntilReminder);
}

// Helper function to convert base64 to Uint8Array
// (required for applicationServerKey)
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }

  return outputArray;
}
