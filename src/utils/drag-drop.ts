import { TaskStatus } from '~/types/task';

// Store the currently dragged task ID
export let draggedTaskId: string | null = null;

// Set the dragged task ID
export function setDraggedTaskId(id: string | null) {
  draggedTaskId = id;
}

// Get the dragged task ID
export function getDraggedTaskId(): string | null {
  return draggedTaskId;
}

// Handle drag start
export function handleDragStart(event: DragEvent, taskId: string) {
  if (!event.dataTransfer) return;
  
  // Set the dragged task ID
  setDraggedTaskId(taskId);
  
  // Set the drag data
  event.dataTransfer.setData('text/plain', taskId);
  event.dataTransfer.effectAllowed = 'move';
  
  // Add a class to the dragged element
  const element = event.target as HTMLElement;
  if (element) {
    setTimeout(() => {
      element.classList.add('opacity-50');
    }, 0);
  }
}

// Handle drag end
export function handleDragEnd(event: DragEvent) {
  // Clear the dragged task ID
  setDraggedTaskId(null);
  
  // Remove the class from the dragged element
  const element = event.target as HTMLElement;
  if (element) {
    element.classList.remove('opacity-50');
  }
}

// Handle drag over
export function handleDragOver(event: DragEvent) {
  // Prevent default to allow drop
  event.preventDefault();
  
  // Set the drop effect
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move';
  }
}

// Handle drop
export function handleDrop(event: DragEvent, status: TaskStatus, onStatusChange: (taskId: string, status: TaskStatus) => void) {
  // Prevent default behavior
  event.preventDefault();
  
  // Get the task ID from the drag data
  const taskId = event.dataTransfer?.getData('text/plain');
  
  // If we have a task ID and it's different from the current status, update it
  if (taskId) {
    onStatusChange(taskId, status);
  }
  
  // Clear the dragged task ID
  setDraggedTaskId(null);
}
