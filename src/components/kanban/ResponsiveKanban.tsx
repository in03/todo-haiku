import { component$, useVisibleTask$, useSignal, $, useStore, noSerialize } from '@builder.io/qwik';
import Sortable from 'sortablejs';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Select } from '~/components/ui';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';
import { getAllTodos, updateTodo, deleteTodo } from '~/services/yjs-sync';
import { SimpleTaskModal } from './SimpleTaskModal';

export const ResponsiveKanban = component$(() => {
  const tasks = useStore<Task[]>([]);
  const isLoading = useSignal(true);
  const refreshTrigger = useSignal(0);
  const sortableInstances = useStore<Record<string, any>>({
    'open': null,
    'doing': null,
    'done': null,
    'blocked': null
  });

  // Modal state
  const modalOpen = useSignal(false);
  const editTask = useSignal<Task | null>(null);
  const defaultStatus = useSignal<TaskStatus>('open');

  // Mobile view state
  const isMobileView = useSignal(false);
  const selectedStatusFilter = useSignal<TaskStatus | 'all'>('all');

  // Scroll state for arrows
  const canScrollLeft = useSignal(false);
  const canScrollRight = useSignal(true);
  const containerRef = useSignal<HTMLElement | null>(null);

  // Scroll amount
  const scrollAmount = 300; // Approximately one column width + gap

  // Check if we're on mobile
  useVisibleTask$(() => {
    const checkMobile = () => {
      isMobileView.value = window.innerWidth < 1024;
    };

    checkMobile();

    // Set up event listener for resize
    window.addEventListener('resize', checkMobile);

    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  });

  // Update scroll arrows visibility
  const updateScrollArrows = $(() => {
    if (!containerRef.value) return;

    const container = containerRef.value;
    const isScrollable = container.scrollWidth > container.clientWidth;

    if (isScrollable) {
      canScrollLeft.value = container.scrollLeft > 5;
      canScrollRight.value = container.scrollLeft < (container.scrollWidth - container.clientWidth - 5);
    } else {
      canScrollLeft.value = false;
      canScrollRight.value = false;
    }
  });

  // Handle scroll events
  const handleScroll = $(() => {
    updateScrollArrows();
  });

  // Scroll left
  const scrollLeft = $(() => {
    if (!containerRef.value) return;
    containerRef.value.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
  });

  // Scroll right
  const scrollRight = $(() => {
    if (!containerRef.value) return;
    containerRef.value.scrollBy({ left: scrollAmount, behavior: 'smooth' });
  });

  // Initialize scroll arrows on component mount
  useVisibleTask$(({ track }) => {
    track(() => tasks.length);
    track(() => isLoading.value);

    // Get container reference
    containerRef.value = document.getElementById('kanban-scroll-container');

    // Initial check for arrows
    setTimeout(updateScrollArrows, 100);
    setTimeout(updateScrollArrows, 500);
    setTimeout(updateScrollArrows, 1000);
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
    }, 30000); // Refresh every 30 seconds

    // Clean up the interval when the component is unmounted
    cleanup(() => clearInterval(intervalId));
  });

  // Initialize Sortable.js for each column (desktop only)
  useVisibleTask$(({ track, cleanup }) => {
    track(() => tasks.length);
    track(() => isLoading.value);
    track(() => isMobileView.value);

    // Only initialize Sortable when loading is complete, tasks are available, and we're on desktop
    if (isLoading.value || isMobileView.value) {
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

  // Get filtered tasks for mobile view
  const getFilteredTasks = () => {
    if (selectedStatusFilter.value === 'all') {
      return tasks;
    }
    return tasks.filter(task => task.status === selectedStatusFilter.value);
  };

  // Open modal to create a new task
  const openCreateModal = $((status: TaskStatus) => {
    editTask.value = null;
    defaultStatus.value = status;
    modalOpen.value = true;
  });

  // Open modal to edit a task
  const openEditModal = $((task: Task) => {
    console.log('Opening edit modal with task:', task);
    editTask.value = { ...task }; // Create a copy of the task
    modalOpen.value = true;
  });

  // Close modal
  const closeModal = $(() => {
    modalOpen.value = false;
    // Refresh tasks after modal closes
    refreshTrigger.value++;
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

  // Status badge colors
  const statusBadgeColors = {
    'open': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    'doing': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300',
    'done': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    'blocked': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
  };

  return (
    <>
      <div class="zen-container">
        {isLoading.value ? (
          <div class="flex justify-center items-center h-64">
            <div class="text-lg text-muted-foreground">Loading tasks...</div>
          </div>
        ) : (
          <>
            {/* Mobile Status Filter */}
            <div class="lg:hidden mb-4">
              <div class="flex items-center justify-between mb-4">
                <Select
                  value={selectedStatusFilter.value}
                  onChange$={(e: any) => selectedStatusFilter.value = e.target.value}
                  class="w-48"
                >
                  <option value="all">All Tasks</option>
                  {Object.entries(TASK_STATUS_CONFIG).map(([value, config]) => (
                    <option key={value} value={value}>
                      {config.label}
                    </option>
                  ))}
                </Select>

                <Button onClick$={() => openCreateModal('open')}>
                  Add Task
                </Button>
              </div>

              {/* Mobile Task List */}
              <div class="space-y-3">
                {getFilteredTasks().map(task => (
                  <div
                    key={task.id}
                    class={`border rounded-md p-4 bg-card shadow-sm cursor-pointer hover:shadow-md transition-all duration-200 border-l-4 ${statusColors[task.status] ? statusColors[task.status].replace('border-t-4 border-t-', 'border-l-') : 'border-l-gray-500'}`}
                    onClick$={() => openEditModal(task)}
                  >
                    <div class="flex items-center mb-2">
                      <span class="mr-2">{statusIcons[task.status] || '📝'}</span>
                      <h3 class="text-lg font-medium">{task.title}</h3>
                    </div>
                    <div class="haiku-text whitespace-pre-line text-sm mt-3 p-2 bg-muted/10 rounded-md">
                      {task.content}
                    </div>
                    <div class="flex justify-between items-center mt-2">
                      <div class="text-xs text-muted-foreground flex items-center">
                        <span class="mr-2">📅</span>
                        {new Date(task.created_at).toLocaleDateString()}
                      </div>
                      <div class="text-xs text-muted-foreground">
                        {TASK_STATUS_CONFIG[task.status]?.description || 'Unknown Status'}
                      </div>
                    </div>
                  </div>
                ))}

                {getFilteredTasks().length === 0 && (
                  <div class="text-center p-8 border border-dashed rounded-md">
                    <p class="text-muted-foreground">No tasks found</p>
                    <Button
                      variant="outline"
                      class="mt-4"
                      onClick$={() => openCreateModal('open')}
                    >
                      Create your first task
                    </Button>
                  </div>
                )}
              </div>
            </div>

            {/* Desktop Kanban Board */}
            <div class="hidden lg:block">
              <div class="relative">
                {/* Left fade effect */}
                <div class="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none"></div>

                {/* Left scroll arrow - only shown when can scroll left */}
                {canScrollLeft.value && (
                  <button
                    onClick$={scrollLeft}
                    class="absolute left-2 top-1/2 transform -translate-y-1/2 bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-20 transition-all duration-300 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    aria-label="Scroll left"
                  >
                    <span class="text-2xl">←</span>
                  </button>
                )}

                {/* Right fade effect */}
                <div class="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none"></div>

                {/* Right scroll arrow - only shown when can scroll right */}
                {canScrollRight.value && (
                  <button
                    onClick$={scrollRight}
                    class="absolute right-2 top-1/2 transform -translate-y-1/2 bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-20 transition-all duration-300 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    aria-label="Scroll right"
                  >
                    <span class="text-2xl">→</span>
                  </button>
                )}

                <div
                  class="overflow-x-auto pb-4 kanban-scroll-container"
                  id="kanban-scroll-container"
                  onScroll$={handleScroll}>
                  <div class="flex gap-6 px-4">
                  {Object.entries(TASK_STATUS_CONFIG).map(([status, config]) => (
                    <Card key={status} class={`h-full flex flex-col ${statusColors[status as TaskStatus]} min-w-[260px] max-w-[260px]`}>
                      <CardHeader class="pb-2 px-4">
                        <div class="flex items-center mb-2">
                          <span class="text-xl mr-2">{statusIcons[status as TaskStatus]}</span>
                          <CardTitle>{config.label}</CardTitle>
                        </div>
                        <CardDescription class="text-xs">{config.description}</CardDescription>
                        <div class="mt-3 p-3 bg-muted/20 rounded-md text-xs italic text-muted-foreground haiku-text whitespace-pre-line font-serif">
                          {config.haiku}
                        </div>
                      </CardHeader>
                      <CardContent class="flex-1 overflow-y-auto max-h-[550px] min-h-[250px] px-4">
                        <div
                          id={`kanban-column-${status}`}
                          data-status={status}
                          class="space-y-4 min-h-[100px]"
                        >
                          {/* Add Task Placeholder - Always at the top */}
                          <div
                            class="task-card border-2 border-dashed rounded-md p-3 bg-card/30 hover:bg-card/60 cursor-pointer transition-all duration-200 text-center"
                            onClick$={() => openCreateModal(status as TaskStatus)}
                          >
                            <div class="flex items-center justify-center py-2">
                              <span class="text-muted-foreground">+ Add a new task</span>
                            </div>
                          </div>

                          {/* Task Cards */}
                          {getTasksByStatus(status as TaskStatus).map(task => (
                            <div
                              key={task.id}
                              class={`task-card border rounded-md p-4 bg-card shadow-sm cursor-move hover:shadow-md transition-all duration-200 border-l-4 ${statusColors[task.status] ? statusColors[task.status].replace('border-t-4 border-t-', 'border-l-') : 'border-l-gray-500'}`}
                              data-task-id={task.id}
                              onClick$={(e) => {
                                // Only open edit modal if not dragging
                                const target = e.target as HTMLElement;
                                if (!target.closest('.drag-handle')) {
                                  openEditModal(task);
                                }
                              }}
                            >
                              <div class="flex items-start mb-2">
                                <div class="flex items-center">
                                  <span class="text-muted-foreground mr-2 drag-handle">⋮⋮</span>
                                  <h3 class="text-lg font-medium">{task.title}</h3>
                                </div>
                              </div>
                              <div class="haiku-text whitespace-pre-line text-sm mt-4 p-3 bg-muted/10 rounded-md font-serif">
                                {task.content}
                              </div>
                              <div class="text-xs text-muted-foreground mt-2 flex items-center">
                                <span class="mr-2">📅</span>
                                {new Date(task.created_at).toLocaleDateString()}
                              </div>
                            </div>
                          ))}

                          {getTasksByStatus(status as TaskStatus).length === 0 && (
                            <div class="text-center p-4 mt-4">
                              <p class="text-muted-foreground text-sm">No tasks yet</p>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Task Modal */}
      <SimpleTaskModal
        isOpen={modalOpen.value}
        onClose={closeModal}
        editTask={editTask.value}
        defaultStatus={defaultStatus.value}
      />
    </>
  );
});
