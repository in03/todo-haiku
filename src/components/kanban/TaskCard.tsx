import { component$, $, useSignal } from '@builder.io/qwik';
import { Card, CardContent, CardFooter, Button, Select } from '~/components/ui';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';
import { deleteTodo } from '~/services/yjs-sync';
import { handleDragStart, handleDragEnd } from '~/utils/drag-drop';

interface TaskCardProps {
  task: Task;
  onStatusChange: (taskId: string, newStatus: TaskStatus) => void;
}

export const TaskCard = component$<TaskCardProps>(({ task, onStatusChange }) => {
  const isExpanded = useSignal(false);
  const isDragging = useSignal(false);

  // Status color mapping
  const statusColors = {
    'open': 'border-l-4 border-l-blue-500',
    'doing': 'border-l-4 border-l-amber-500',
    'done': 'border-l-4 border-l-green-500',
    'blocked': 'border-l-4 border-l-red-500'
  };

  // Handle status change
  const handleStatusChange = $((e: any) => {
    const newStatus = e.target.value as TaskStatus;
    onStatusChange(task.id, newStatus);
  });

  // Handle delete
  const handleDelete = $(() => {
    if (confirm('Are you sure you want to delete this task?')) {
      deleteTodo(task.id);
    }
  });

  // Toggle expanded state
  const toggleExpanded = $(() => {
    isExpanded.value = !isExpanded.value;
  });

  // Handle drag start
  const onDragStart = $((e: any) => {
    handleDragStart(e, task.id);
    isDragging.value = true;
  });

  // Handle drag end
  const onDragEnd = $((e: any) => {
    handleDragEnd(e);
    isDragging.value = false;
  });

  return (
    <Card
      class={`transition-all duration-200 hover:shadow-md cursor-pointer ${statusColors[task.status] || 'border-l-4 border-l-gray-500'} mb-3 ${isDragging.value ? 'dragging' : ''}`}
      onClick$={toggleExpanded}
      draggable={true}
      onDragStart$={onDragStart}
      onDragEnd$={onDragEnd}
    >
      <CardContent class="p-4">
        <div class="flex justify-between items-start mb-2">
          <h3 class="text-lg font-medium">{task.title}</h3>
          <div class="text-xs px-2 py-1 rounded-full bg-muted">
            {TASK_STATUS_CONFIG[task.status]?.label || 'Unknown'}
          </div>
        </div>

        {isExpanded.value ? (
          <div class="haiku-text whitespace-pre-line text-sm mt-3 p-2 bg-muted/20 rounded-md">
            {task.content}
          </div>
        ) : (
          <div class="haiku-text whitespace-pre-line text-sm mt-3 line-clamp-2 text-muted-foreground">
            {task.content.split('\n')[0]}...
          </div>
        )}

        <div class="text-xs text-muted-foreground mt-2 flex items-center">
          <span class="mr-2">📅</span>
          {new Date(task.created_at).toLocaleDateString()}
        </div>
      </CardContent>

      {isExpanded.value && (
        <CardFooter class="flex justify-between p-4 pt-0 border-t border-border mt-2">
          <Select
            value={task.status}
            onChange$={handleStatusChange}
            onClick$={(e) => e.stopPropagation()}
            class="text-xs w-40"
          >
            {Object.entries(TASK_STATUS_CONFIG).map(([value, config]) => {
              return (
                <option key={value} value={value}>
                  {config.label}
                </option>
              );
            })}
          </Select>

          <Button
            variant="destructive"
            size="sm"
            onClick$={(e) => {
              e.stopPropagation();
              handleDelete();
            }}
            class="text-xs"
          >
            Delete
          </Button>
        </CardFooter>
      )}
    </Card>
  );
});
