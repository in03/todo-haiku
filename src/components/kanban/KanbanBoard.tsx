import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import { TaskCard } from './TaskCard';
import { KanbanColumn } from './KanbanColumn';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';
import { getAllTodos, updateTodo } from '~/services/yjs-sync';

export const KanbanBoard = component$(() => {
  const tasks = useStore<Task[]>([]);
  const isLoading = useSignal(true);

  // Create a signal to trigger task refresh
  const refreshTrigger = useSignal(0);

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
    }, 2000); // Refresh every 2 seconds

    // Clean up the interval when the component is unmounted
    cleanup(() => clearInterval(intervalId));
  });

  // Handle task status change
  const handleStatusChange = $((taskId: string, newStatus: TaskStatus) => {
    // Update task in Y.js
    updateTodo(taskId, { status: newStatus });

    // Update local state
    const taskIndex = tasks.findIndex(task => task.id === taskId);
    if (taskIndex !== -1) {
      tasks[taskIndex].status = newStatus;
    }
  });

  // Group tasks by status
  const getTasksByStatus = (status: TaskStatus) => {
    return tasks.filter(task => task.status === status);
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
              <KanbanColumn
                key={status}
                title={config.label}
                description={config.description}
                haiku={config.haiku}
                status={status as TaskStatus}
                onStatusChange={handleStatusChange}
              >
                {getTasksByStatus(status as TaskStatus).map(task => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    onStatusChange={handleStatusChange}
                  />
                ))}
              </KanbanColumn>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
