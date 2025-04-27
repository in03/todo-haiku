import { component$, useVisibleTask$, useSignal, $, useStore, noSerialize, useContextProvider } from '@builder.io/qwik';
import Sortable from 'sortablejs';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Select } from '~/components/ui';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';
import { getAllTodos, updateTodo } from '~/services/yjs-sync';
import { SimpleTaskModal } from './SimpleTaskModal';
import { TaskCard } from './TaskCard';
import { KanbanOptions, SortOption } from './KanbanOptions';
import { KanbanContext, KanbanContextState } from './KanbanContext';
import { filterTasksBySearchTerm, sortTasks } from '~/utils/search';

export const ResponsiveKanban = component$(() => {
  const tasks = useStore<Task[]>([]);
  const filteredTasks = useStore<Task[]>([]);
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

  // Kanban options state
  const kanbanState = useStore<KanbanContextState>({
    searchTerm: '',
    sortOption: 'custom',
    quietMode: false
  });

  // Provide the kanban context
  useContextProvider(KanbanContext, kanbanState);

  // Scroll state for arrows
  const canScrollLeft = useSignal(false);
  const canScrollRight = useSignal(true);
  const containerRef = useSignal<HTMLElement | null>(null);
  const windowScrollY = useSignal(0);
  const isKanbanInView = useSignal(true);
  const arrowsSticky = useSignal(false);
  const kanbanTopPosition = useSignal(0);

  // Scroll amount
  const scrollAmount = 300; // Approximately one column width + gap

  // Check if we're on mobile and track window scroll position
  useVisibleTask$(() => {
    const checkMobile = () => {
      isMobileView.value = window.innerWidth < 1024;
    };

    const updateScrollPosition = () => {
      windowScrollY.value = window.scrollY;

      // Check if kanban board is in view and calculate sticky state
      if (containerRef.value) {
        const rect = containerRef.value.getBoundingClientRect();
        const kanbanContainer = document.querySelector('.hidden.lg\\:block');

        // Consider it in view if any part of it is visible in the viewport
        isKanbanInView.value = (
          rect.top < window.innerHeight &&
          rect.bottom > 0
        );

        // Store the top position of the kanban board relative to the page
        if (kanbanContainer) {
          const kanbanRect = kanbanContainer.getBoundingClientRect();
          kanbanTopPosition.value = kanbanRect.top + window.scrollY;

          // Calculate if arrows should be fixed to the viewport
          // They become fixed when the kanban is scrolled enough that we want the arrows
          // to follow the viewport instead of staying with the kanban

          // Get the kanban's position relative to the viewport
          const kanbanTopRelativeToViewport = kanbanRect.top;

          // We want the arrows to become sticky when the kanban's top edge
          // is about to leave the viewport (with a small buffer)
          const stickyThreshold = 100; // 100px buffer

          // If the kanban's top is above this threshold, make the arrows sticky
          arrowsSticky.value = kanbanTopRelativeToViewport < stickyThreshold;
        }
      }
    };

    checkMobile();
    updateScrollPosition();

    // Set up event listeners
    window.addEventListener('resize', checkMobile);
    window.addEventListener('scroll', updateScrollPosition);

    return () => {
      window.removeEventListener('resize', checkMobile);
      window.removeEventListener('scroll', updateScrollPosition);
    };
  });

  // Update scroll arrows visibility
  const updateScrollArrows = $(() => {
    if (!containerRef.value) return;

    const container = containerRef.value;
    const isScrollable = container.scrollWidth > container.clientWidth;

    // Always show arrows if the container is scrollable
    // This ensures arrows are visible before any scrolling happens
    if (isScrollable) {
      // Left arrow is visible if we can scroll left (not at the beginning)
      canScrollLeft.value = container.scrollLeft > 5;

      // Right arrow is always visible if the content is wider than the container
      canScrollRight.value = true;
    } else {
      canScrollLeft.value = false;
      canScrollRight.value = false;
    }
  });

  // Handle horizontal scroll events
  const handleScroll = $(() => {
    updateScrollArrows();
  });

  // Track window scroll to update arrow visibility
  useVisibleTask$(({ track }) => {
    track(() => windowScrollY.value);
    track(() => isKanbanInView.value);

    // Update arrows when window is scrolled or kanban visibility changes
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

    // Initial check for arrows and position
    setTimeout(() => {
      // Force the right arrow to be visible initially if content is scrollable
      if (containerRef.value && containerRef.value.scrollWidth > containerRef.value.clientWidth) {
        canScrollRight.value = true;
      }

      updateScrollArrows();

      // Get initial kanban position
      const kanbanContainer = document.querySelector('.hidden.lg\\:block');
      if (kanbanContainer) {
        const kanbanRect = kanbanContainer.getBoundingClientRect();
        kanbanTopPosition.value = kanbanRect.top + window.scrollY;

        // Calculate initial sticky state using the same logic as in updateScrollPosition

        // Get the kanban's position relative to the viewport
        const kanbanTopRelativeToViewport = kanbanRect.top;

        // We want the arrows to become sticky when the kanban's top edge
        // is about to leave the viewport (with a small buffer)
        const stickyThreshold = 100; // 100px buffer

        // If the kanban's top is above this threshold, make the arrows sticky
        arrowsSticky.value = kanbanTopRelativeToViewport < stickyThreshold;
      }
    }, 100);

    // Multiple checks to ensure arrows are properly initialized
    setTimeout(updateScrollArrows, 300);
    setTimeout(updateScrollArrows, 500);
    setTimeout(updateScrollArrows, 1000);
  });

  // Load tasks on component mount and when refreshTrigger changes
  useVisibleTask$(({ track }) => {
    console.log('Task loading useVisibleTask$ running');

    // Track the refresh trigger to reload tasks when it changes
    track(() => refreshTrigger.value);
    console.log('Refresh trigger value:', refreshTrigger.value);

    // Also track search and sort options
    track(() => kanbanState.searchTerm);
    track(() => kanbanState.sortOption);
    console.log('Search term:', kanbanState.searchTerm);
    console.log('Sort option:', kanbanState.sortOption);

    try {
      console.log('Loading tasks...');
      // Get tasks from Y.js
      let allTasks = getAllTodos();
      console.log('All tasks:', allTasks);

      // Check if this is a new account/initial state
      // We'll use localStorage to track if we've shown sample tasks before
      const hasShownSampleTasks = localStorage.getItem('hasShownSampleTasks') === 'true';

      // Only create sample tasks if no tasks are found AND we haven't shown sample tasks before
      if (allTasks.length === 0 && !hasShownSampleTasks) {
        console.log('No tasks found and no sample tasks shown before, creating sample tasks');

        // Create sample tasks
        const sampleTasks = [
          {
            id: 'sample-1',
            title: 'Sample Task 1',
            content: 'Autumn leaves falling\nGently dancing in the breeze\nNature\'s lullaby',
            status: 'open',
            is_completed: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            user_id: 'local',
            sortOrder: 0
          },
          {
            id: 'sample-2',
            title: 'Sample Task 2',
            content: 'Mountain silhouette\nStanding tall against the sky\nTimeless sentinel',
            status: 'doing',
            is_completed: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            user_id: 'local',
            sortOrder: 1
          },
          {
            id: 'sample-3',
            title: 'Sample Task 3',
            content: 'Ocean waves crashing\nWhispering ancient secrets\nEternal rhythm',
            status: 'done',
            is_completed: true,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            user_id: 'local',
            sortOrder: 2
          }
        ];

        // Add sample tasks to the array (not to Y.js storage)
        // This will only affect the current session
        sampleTasks.forEach(task => {
          allTasks.push(task);
        });

        // Mark that we've shown sample tasks
        localStorage.setItem('hasShownSampleTasks', 'true');

        console.log('Added sample tasks:', allTasks.length, 'total tasks');
      } else if (allTasks.length === 0) {
        console.log('No tasks found, but sample tasks have been shown before. Respecting user\'s choice to have no tasks.');
      } else {
        console.log('Found existing tasks:', allTasks.length, 'tasks');
      }

      // Convert old todos to new task format if needed
      const convertedTasks = allTasks.map((todo: any) => {
        // If the todo doesn't have a status field, add it
        if (!todo.status) {
          return {
            ...todo,
            status: todo.is_completed ? 'done' : 'open'
          };
        }

        // Ensure sortOrder exists
        if (todo.sortOrder === undefined) {
          todo.sortOrder = 0;
        }

        return todo;
      });

      // Update tasks store
      tasks.length = 0;
      tasks.push(...convertedTasks);

      console.log('Loaded tasks:', tasks.length, 'tasks');

      // Apply search filter and sorting
      updateFilteredTasks();

      // Force update filtered tasks with all tasks if it's empty
      if (filteredTasks.length === 0 && tasks.length > 0) {
        console.log('Filtered tasks is empty but tasks has items, forcing update');
        filteredTasks.length = 0;
        filteredTasks.push(...tasks);
      }

      console.log('Final filtered tasks:', filteredTasks.length, 'tasks');
      isLoading.value = false;
    } catch (error) {
      console.error('Error loading tasks:', error);
      isLoading.value = false;
    }
  });

  // Function to update filtered tasks based on search and sort options
  const updateFilteredTasks = $(() => {
    console.log('updateFilteredTasks called with', tasks.length, 'tasks');

    // First filter by search term
    let filtered = filterTasksBySearchTerm(tasks, kanbanState.searchTerm);

    // Then sort according to the selected option
    filtered = sortTasks(filtered, kanbanState.sortOption);

    // Update filtered tasks
    filteredTasks.length = 0;
    filteredTasks.push(...filtered);

    console.log('Updated filtered tasks:', filteredTasks.length, 'tasks');

    // If filtered tasks is empty but tasks has items, use all tasks
    if (filteredTasks.length === 0 && tasks.length > 0) {
      console.log('Filtered tasks is empty but tasks has items, using all tasks');
      filteredTasks.length = 0;
      filteredTasks.push(...tasks);
      console.log('Now filtered tasks has', filteredTasks.length, 'tasks');
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
              // No handle option makes the entire card draggable
              // Add delay and fallbackTolerance to better distinguish between clicks and drags
              delay: 150, // Delay before dragging starts (helps distinguish clicks)
              delayOnTouchOnly: true, // Only apply delay for touch devices
              fallbackTolerance: 5, // How many pixels the pointer needs to move to start dragging
              onEnd: (evt) => {
                const taskId = evt.item.getAttribute('data-task-id');
                const newStatus = evt.to.getAttribute('data-status') as TaskStatus;
                const newIndex = evt.newIndex !== undefined ? evt.newIndex : 0;

                if (taskId && newStatus) {
                  // Update task status in Y.js
                  updateTodo(taskId, {
                    status: newStatus,
                    // Update sort order based on position in the list
                    sortOrder: newIndex
                  });

                  // Update local state
                  const taskIndex = tasks.findIndex(task => task.id === taskId);
                  if (taskIndex !== -1) {
                    tasks[taskIndex].status = newStatus;
                    tasks[taskIndex].sortOrder = newIndex;

                    // Update other tasks' sort orders in the same column
                    const tasksInSameColumn = tasks.filter(t => t.status === newStatus && t.id !== taskId);
                    tasksInSameColumn.forEach((task, idx) => {
                      const newOrder = idx >= newIndex ? idx + 1 : idx;
                      task.sortOrder = newOrder;
                      updateTodo(task.id, { sortOrder: newOrder });
                    });

                    // Refresh filtered tasks
                    updateFilteredTasks();
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
    // Check if filteredTasks is empty but tasks has items
    if (filteredTasks.length === 0 && tasks.length > 0) {
      console.log('getTasksByStatus: filteredTasks is empty, using tasks directly for status', status);
      const tasksWithStatus = tasks.filter(task => task.status === status);
      console.log('Found', tasksWithStatus.length, 'tasks with status', status);
      return tasksWithStatus;
    }

    // Use the filteredTasks array that's already filtered and sorted
    // This ensures we're using the same filtered tasks across the app
    const filteredTasksWithStatus = filteredTasks.filter(task => task.status === status);
    console.log('getTasksByStatus: Found', filteredTasksWithStatus.length, 'filtered tasks with status', status);
    return filteredTasksWithStatus;
  };

  // Get filtered tasks for mobile view
  const getFilteredTasks = () => {
    console.log('getFilteredTasks called, filteredTasks:', filteredTasks.length, 'tasks, all tasks:', tasks.length);

    // Check if filteredTasks is empty but tasks has items
    if (filteredTasks.length === 0 && tasks.length > 0) {
      console.log('getFilteredTasks: filteredTasks is empty, using tasks directly');
      let filtered = [...tasks];

      // Apply status filter for mobile view
      if (selectedStatusFilter.value !== 'all') {
        filtered = filtered.filter(task => task.status === selectedStatusFilter.value);
        console.log('Filtered to', filtered.length, 'tasks with status', selectedStatusFilter.value);
      } else {
        console.log('No status filter applied, returning all', filtered.length, 'tasks');
      }

      return filtered;
    }

    // Start with the already filtered and sorted tasks
    let filtered = [...filteredTasks];
    console.log('Using', filtered.length, 'filtered tasks');

    // Then apply status filter for mobile view
    if (selectedStatusFilter.value !== 'all') {
      filtered = filtered.filter(task => task.status === selectedStatusFilter.value);
      console.log('Filtered to', filtered.length, 'tasks with status', selectedStatusFilter.value);
    }

    return filtered;
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
    'open': '🌱',
    'doing': '🧹',
    'done': '🟡',
    'blocked': '🍂'
  };

  // Handle task status change
  const handleStatusChange = $((taskId: string, newStatus: TaskStatus) => {
    // Update task status in Y.js
    updateTodo(taskId, { status: newStatus });

    // Update local state
    const taskIndex = tasks.findIndex(task => task.id === taskId);
    if (taskIndex !== -1) {
      tasks[taskIndex].status = newStatus;

      // Refresh filtered tasks
      updateFilteredTasks();
    }
  });

  // Handle search input change
  const handleSearch = $((searchTerm: string) => {
    kanbanState.searchTerm = searchTerm;
    updateFilteredTasks();
  });

  // Handle sort option change
  const handleSortChange = $((sortOption: SortOption) => {
    kanbanState.sortOption = sortOption;
    updateFilteredTasks();
  });

  // Handle quiet mode toggle
  const handleQuietModeToggle = $((enabled: boolean) => {
    kanbanState.quietMode = enabled;

    // Reset all task twirl states when toggling quiet mode off
    if (!enabled) {
      // Update all tasks to have isCollapsed = false
      tasks.forEach(task => {
        task.isCollapsed = false;
        updateTodo(task.id, { isCollapsed: false });
      });
    }
  });

  return (
    <>
      <div class="zen-container">
        {isLoading.value ? (
          <div class="flex justify-center items-center h-64">
            <div class="text-lg text-muted-foreground">Loading tasks...</div>
          </div>
        ) : (
          <>
            {/* Kanban Options Bar */}
            <KanbanOptions
              onSearch={handleSearch}
              onSortChange={handleSortChange}
              onQuietModeToggle={handleQuietModeToggle}
            />

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
                      <button
                        onClick$={(e) => {
                          e.stopPropagation();
                          if (!kanbanState.quietMode) {
                            const newCollapsedState = !(task.isCollapsed !== false);
                            updateTodo(task.id, { isCollapsed: newCollapsedState });
                            // Update local state
                            const taskIndex = tasks.findIndex(t => t.id === task.id);
                            if (taskIndex !== -1) {
                              tasks[taskIndex].isCollapsed = newCollapsedState;
                            }
                          }
                        }}
                        class="mr-2 text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                      >
                        <span class={`inline-block transition-transform duration-200 text-xs ${(kanbanState.quietMode || task.isCollapsed) ? '' : 'rotate-90'}`}>
                          ▶
                        </span>
                      </button>
                      <h3 class="text-lg font-medium">{task.title}</h3>
                    </div>
                    {!kanbanState.quietMode && task.isCollapsed !== true && (
                      <div class="haiku-text whitespace-pre-line text-sm mt-3 p-2 bg-muted/10 rounded-md">
                        {task.content}
                      </div>
                    )}
                    <div class="text-xs text-muted-foreground mt-2 flex items-center">
                      <span class="mr-2">📅</span>
                      {new Date(task.created_at).toLocaleDateString()}
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

                {/* Left scroll arrow - always fixed position */}
                {canScrollLeft.value && isKanbanInView.value && (
                  <button
                    id="left-scroll-arrow"
                    onClick$={scrollLeft}
                    class="bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-50 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    style={{
                      position: 'fixed', // Always fixed position
                      left: '1rem',
                      top: '50%', // Centered in viewport when sticky
                      transform: 'translateY(-50%)',
                      opacity: arrowsSticky.value ? 1 : 0, // Only visible when sticky
                      pointerEvents: arrowsSticky.value ? 'auto' : 'none' // Only clickable when sticky
                    }}
                    aria-label="Scroll left"
                  >
                    <span class="text-2xl">←</span>
                  </button>
                )}

                {/* Left scroll arrow - always absolute position */}
                {canScrollLeft.value && isKanbanInView.value && (
                  <button
                    id="left-scroll-arrow-absolute"
                    onClick$={scrollLeft}
                    class="bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-50 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    style={{
                      position: 'absolute', // Always absolute position
                      left: '1rem',
                      top: '200px', // Static offset from top of container
                      transform: 'none', // No transform needed
                      opacity: arrowsSticky.value ? 0 : 1, // Only visible when not sticky
                      pointerEvents: arrowsSticky.value ? 'none' : 'auto' // Only clickable when not sticky
                    }}
                    aria-label="Scroll left"
                  >
                    <span class="text-2xl">←</span>
                  </button>
                )}

                {/* Right fade effect */}
                <div class="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none"></div>

                {/* Right scroll arrow - always fixed position */}
                {canScrollRight.value && isKanbanInView.value && (
                  <button
                    id="right-scroll-arrow"
                    onClick$={scrollRight}
                    class="bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-50 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    style={{
                      position: 'fixed', // Always fixed position
                      right: '1rem',
                      top: '50%', // Centered in viewport when sticky
                      transform: 'translateY(-50%)',
                      opacity: arrowsSticky.value ? 1 : 0, // Only visible when sticky
                      pointerEvents: arrowsSticky.value ? 'auto' : 'none' // Only clickable when sticky
                    }}
                    aria-label="Scroll right"
                  >
                    <span class="text-2xl">→</span>
                  </button>
                )}

                {/* Right scroll arrow - always absolute position */}
                {canScrollRight.value && isKanbanInView.value && (
                  <button
                    id="right-scroll-arrow-absolute"
                    onClick$={scrollRight}
                    class="bg-background/90 text-foreground rounded-full w-12 h-12 flex items-center justify-center shadow-lg cursor-pointer z-50 hover:bg-primary/10 hover:border-primary border-2 border-border"
                    style={{
                      position: 'absolute', // Always absolute position
                      right: '1rem',
                      top: '200px', // Static offset from top of container
                      transform: 'none', // No transform needed
                      opacity: arrowsSticky.value ? 0 : 1, // Only visible when not sticky
                      pointerEvents: arrowsSticky.value ? 'none' : 'auto' // Only clickable when not sticky
                    }}
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
                      {/* Remove max-height constraint to allow full vertical scrolling */}
                      <CardContent class="flex-1 px-4">
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
                          {getTasksByStatus(status as TaskStatus)
                            .map(task => (
                              <TaskCard
                                key={task.id}
                                task={task}
                                onStatusChange={handleStatusChange}
                                onEdit={openEditModal}
                                forceCollapsed={kanbanState.quietMode}
                              />
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
