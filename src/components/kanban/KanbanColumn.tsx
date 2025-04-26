import { component$, Slot, $, useSignal } from '@builder.io/qwik';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '~/components/ui';
import { TaskStatus } from '~/types/task';
import { handleDragOver, handleDrop } from '~/utils/drag-drop';

interface KanbanColumnProps {
  title: string;
  description: string;
  haiku: string;
  status: TaskStatus;
  onStatusChange?: (taskId: string, status: TaskStatus) => void;
}

export const KanbanColumn = component$<KanbanColumnProps>(({ title, description, haiku, status, onStatusChange }) => {
  // Status color mapping
  const statusColors = {
    'open': 'border-t-4 border-t-blue-500',
    'doing': 'border-t-4 border-t-amber-500',
    'done': 'border-t-4 border-t-green-500',
    'blocked': 'border-t-4 border-t-red-500'
  };

  // Status icons
  const statusIcons = {
    'open': '🌱',
    'doing': '🧹',
    'done': '🟡',
    'blocked': '🍂'
  };

  // Track if we're dragging over this column
  const isDragOver = useSignal(false);

  // Handle drag over
  const onDragOver = $((e: any) => {
    handleDragOver(e);
    isDragOver.value = true;
  });

  // Handle drag leave
  const onDragLeave = $(() => {
    isDragOver.value = false;
  });

  // Handle drop
  const onDrop = $((e: any) => {
    if (onStatusChange) {
      handleDrop(e, status, onStatusChange);
    }
    isDragOver.value = false;
  });

  return (
    <Card
      class={`h-full flex flex-col ${statusColors[status]} ${isDragOver.value ? 'drag-over' : ''}`}
      onDragOver$={onDragOver}
      onDragLeave$={onDragLeave}
      onDrop$={onDrop}
    >
      <CardHeader>
        <div class="flex items-center mb-2">
          <span class="text-xl mr-2">{statusIcons[status]}</span>
          <CardTitle>{title}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
        <div class="mt-3 p-3 bg-muted/20 rounded-md text-xs italic text-muted-foreground haiku-text whitespace-pre-line">
          {haiku}
        </div>
      </CardHeader>
      <CardContent class="flex-1 overflow-y-auto max-h-[500px] min-h-[200px]">
        <div class="space-y-3">
          <Slot />
        </div>
        {/* Drop zone indicator - only show when dragging over */}
        {isDragOver.value && (
          <div class="mt-4 border-2 border-dashed border-accent rounded-md p-4 text-center text-accent animate-pulse">
            <p>Drop task here to move to {title}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
});
