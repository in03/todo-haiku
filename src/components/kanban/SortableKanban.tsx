import { component$, useVisibleTask$, useSignal, $, useStore, noSerialize } from '@builder.io/qwik';
import Sortable from 'sortablejs';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button } from '~/components/ui';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';
import { getAllTodos, updateTodo, deleteTodo } from '~/services/yjs-sync';

export const SortableKanban = component$(() => {
  const tasks = useStore<Task[]>([]);
  const isLoading = useSignal(true);
  const refreshTrigger = useSignal(0);
  const sortableInstances = useStore<Record<string, any>>({
    'open': null,
    'doing': null,
    'done': null,
    'blocked': null
  });

  // Load tasks on component mount and when refreshTrigger changes
  useVisibleTask$(({ track }) => {
    // Track the refresh trigger to reload tasks when it changes
    track(() => refreshTrigger.value);

    try {
      console.log('Loading tasks...');
      // Get tasks from Y.js
      const allTasks = getAllTodos();
      console.log('All tasks:', allTasks);

      // Convert old todos to new task format if needed
      const convertedTasks = allTasks.map((todo: any) => {
        // If the todo doesn't have a status field, add it
        if (!todo.status) {
          return {
            ...todo,
            status: todo.is_completed ? 'done' : 'open'
          };
        }
        return todo;
      });

      // Update tasks store
      tasks.length = 0;
      tasks.push(...convertedTasks);

      // Sort by creation date (newest first)
      tasks.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      isLoading.value = false;
    } catch (error) {
      console.error('Error loading tasks:', error);
      isLoading.value = false;
    }
  });

  // Set up an interval to refresh tasks periodically
  useVisibleTask$(({ cleanup }) => {
    const intervalId = setInterval(() => {
      refreshTrigger.value++;
    }, 5000); // Refresh every 5 seconds

    // Clean up the interval when the component is unmounted
    cleanup(() => clearInterval(intervalId));
  });

  // Initialize Sortable.js for each column
  useVisibleTask$(({ track, cleanup }) => {
    track(() => tasks.length);
    track(() => isLoading.value);

    // Only initialize Sortable when loading is complete and tasks are available
    if (isLoading.value) {
      return;
    }

    // Use setTimeout to ensure DOM is fully rendered
    setTimeout(() => {
      // Initialize Sortable for each column
      Object.keys(TASK_STATUS_CONFIG).forEach((status) => {
        const el = document.getElementById(`kanban-column-${status}`);
        if (el) {
          // Destroy existing instance if it exists
          if (sortableInstances[status]) {
            try {
              sortableInstances[status]?.destroy();
            } catch (e) {
              console.error('Error destroying previous Sortable instance:', e);
            }
          }

          try {
            // Create new Sortable instance with noSerialize to prevent serialization issues
            sortableInstances[status] = noSerialize(Sortable.create(el, {
              group: 'kanban',
              animation: 150,
              ghostClass: 'sortable-ghost',
              chosenClass: 'sortable-chosen',
              dragClass: 'sortable-drag',
              handle: '.drag-handle',
              onEnd: (evt) => {
                const taskId = evt.item.getAttribute('data-task-id');
                const newStatus = evt.to.getAttribute('data-status') as TaskStatus;

                if (taskId && newStatus) {
                  // Update task status in Y.js
                  updateTodo(taskId, { status: newStatus });

                  // Update local state
                  const taskIndex = tasks.findIndex(task => task.id === taskId);
                  if (taskIndex !== -1) {
                    tasks[taskIndex].status = newStatus;
                  }
                }
              }
            }));
          } catch (e) {
            console.error('Error creating Sortable instance:', e);
          }
        } else {
          console.warn(`Element #kanban-column-${status} not found`);
        }
      });
    }, 100); // Small delay to ensure DOM is ready

    // Clean up Sortable instances when component unmounts
    cleanup(() => {
      // Use setTimeout to ensure we're not cleaning up during a render cycle
      setTimeout(() => {
        Object.keys(sortableInstances).forEach(key => {
          const instance = sortableInstances[key];
          if (instance) {
            try {
              instance.destroy();
              sortableInstances[key] = null;
            } catch (e) {
              console.error('Error destroying Sortable instance:', e);
            }
          }
        });
      }, 0);
    });
  });

  // Group tasks by status
  const getTasksByStatus = (status: TaskStatus) => {
    return tasks.filter(task => task.status === status);
  };

  // Handle task deletion
  const handleDeleteTask = $((taskId: string) => {
    if (confirm('Are you sure you want to delete this task?')) {
      deleteTodo(taskId);
      refreshTrigger.value++;
    }
  });

  // Status color mapping
  const statusColors = {
    'open': 'border-t-4 border-t-blue-500',
    'doing': 'border-t-4 border-t-amber-500',
    'done': 'border-t-4 border-t-green-500',
    'blocked': 'border-t-4 border-t-red-500'
  };

  // Status icons
  const statusIcons = {
    'open': '💭',
    'doing': '🧹',
    'done': '🔔',
    'blocked': '☔'
  };

  return (
    <div class="zen-container">
      {isLoading.value ? (
        <div class="flex justify-center items-center h-64">
          <div class="text-lg text-muted-foreground">Loading tasks...</div>
        </div>
      ) : (
        <div class="overflow-x-auto pb-4">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 min-w-[1200px]">
            {Object.entries(TASK_STATUS_CONFIG).map(([status, config]) => (
              <Card key={status} class={`h-full flex flex-col ${statusColors[status as TaskStatus] || 'border-t-4 border-t-gray-500'}`}>
                <CardHeader>
                  <div class="flex items-center mb-2">
                    <span class="text-xl mr-2">{statusIcons[status as TaskStatus] || '📝'}</span>
                    <CardTitle>{config.label}</CardTitle>
                  </div>
                  <CardDescription>{config.description}</CardDescription>
                  <div class="mt-3 p-3 bg-muted/20 rounded-md text-xs italic text-muted-foreground haiku-text whitespace-pre-line">
                    {config.haiku}
                  </div>
                </CardHeader>
                <CardContent class="flex-1 overflow-y-auto max-h-[500px] min-h-[200px]">
                  <div
                    id={`kanban-column-${status}`}
                    data-status={status}
                    class="space-y-3 min-h-[100px]"
                  >
                    {getTasksByStatus(status as TaskStatus).map(task => (
                      <div
                        key={task.id}
                        class="task-card border rounded-md p-4 bg-card shadow-sm cursor-move hover:shadow-md transition-all duration-200"
                        data-task-id={task.id}
                      >
                        <div class="flex justify-between items-start mb-2">
                          <div class="flex items-center">
                            <span class="text-muted-foreground mr-2 drag-handle">⋮⋮</span>
                            <h3 class="text-lg font-medium">{task.title}</h3>
                          </div>
                          <div class="text-xs px-2 py-1 rounded-full bg-muted">
                            {TASK_STATUS_CONFIG[task.status]?.label || 'Unknown'}
                          </div>
                        </div>
                        <div class="haiku-text whitespace-pre-line text-sm mt-3 p-2 bg-muted/10 rounded-md">
                          {task.content}
                        </div>
                        <div class="flex justify-between items-center mt-4">
                          <div class="text-xs text-muted-foreground flex items-center">
                            <span class="mr-2">📅</span>
                            {new Date(task.created_at).toLocaleDateString()}
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick$={() => handleDeleteTask(task.id)}
                            class="text-xs"
                          >
                            Delete
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
