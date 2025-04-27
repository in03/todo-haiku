import { component$, $, useSignal, useContext } from '@builder.io/qwik';
import { Card, CardContent } from '~/components/ui';
import { Task, TaskStatus } from '~/types/task';
import { updateTodo } from '~/services/yjs-sync';
import { handleDragStart, handleDragEnd } from '~/utils/drag-drop';
import { KanbanContext } from './KanbanContext';

interface TaskCardProps {
  task: Task;
  onStatusChange: (taskId: string, newStatus: TaskStatus) => void;
  onEdit?: (task: Task) => void;
  forceCollapsed?: boolean;
}

export const TaskCard = component$<TaskCardProps>(({ task, onEdit, forceCollapsed = false }) => {
  const isCollapsed = useSignal(task.isCollapsed === true);
  const isDragging = useSignal(false);
  const kanbanContext = useContext(KanbanContext);

  // Status color mapping
  const statusColors = {
    'open': 'border-l-4 border-l-blue-500',
    'doing': 'border-l-4 border-l-amber-500',
    'done': 'border-l-4 border-l-green-500',
    'blocked': 'border-l-4 border-l-red-500'
  };

  // Toggle collapsed state
  const toggleCollapsed = $((e: MouseEvent) => {
    e.stopPropagation();

    // Only allow toggling if not in quiet mode
    if (!kanbanContext?.quietMode) {
      const newCollapsedState = !isCollapsed.value;
      isCollapsed.value = newCollapsedState;

      // Update the task in the database
      updateTodo(task.id, {
        isCollapsed: newCollapsedState
      });
    }
  });

  // Handle click to edit
  const handleClick = $((e: MouseEvent) => {
    // If not dragging and onEdit is provided, call it
    if (!isDragging.value && onEdit) {
      onEdit(task);
    }
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

  // Determine if content should be collapsed
  const shouldCollapseContent = forceCollapsed ||
    (kanbanContext?.quietMode) ||
    isCollapsed.value;

  return (
    <Card
      class={`transition-all duration-200 hover:shadow-md cursor-move ${statusColors[task.status] || 'border-l-4 border-l-gray-500'} mb-3 ${isDragging.value ? 'dragging' : ''}`}
      draggable={true}
      onDragStart$={onDragStart}
      onDragEnd$={onDragEnd}
      onClick$={handleClick}
    >
      <CardContent class="p-4">
        <div class="flex items-start mb-2">
          <div class="flex items-center">
            <button
              onClick$={toggleCollapsed}
              class="mr-2 text-muted-foreground/50 hover:text-muted-foreground transition-colors"
              aria-label={isCollapsed.value ? "Expand task" : "Collapse task"}
            >
              <span class={`inline-block transition-transform duration-200 text-xs ${shouldCollapseContent ? '' : 'rotate-90'}`}>
                ▶
              </span>
            </button>
            <h3 class="text-lg font-medium">{task.title}</h3>
          </div>
        </div>

        {!shouldCollapseContent && (
          <div class="haiku-text whitespace-pre-line text-sm mt-3 p-2 bg-muted/20 rounded-md">
            {task.content}
          </div>
        )}

        <div class="text-xs text-muted-foreground mt-2 flex items-center">
          <span class="mr-2">📅</span>
          {new Date(task.created_at).toLocaleDateString()}
        </div>
      </CardContent>
    </Card>
  );
});
