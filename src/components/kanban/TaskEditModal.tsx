import { component$, $, useSignal, useStore, useVisibleTask$ } from '@builder.io/qwik';
import {
  Button,
  Input,
  Label,
  Textarea,
  Select,
  Modal,
  ModalFooter
} from '~/components/ui';
import { validateHaiku, generateHaikuTemplate } from '~/utils/haiku-validator';
import { updateTodo, deleteTodo } from '~/services/yjs-sync';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';

interface TaskEditModalProps {
  open: boolean;
  onClose: () => void;
  taskId: string;
  taskTitle: string;
  taskContent: string;
  taskStatus: TaskStatus;
}

export const TaskEditModal = component$<TaskEditModalProps>(({
  open,
  onClose,
  taskId,
  taskTitle,
  taskContent,
  taskStatus
}) => {
  console.log('TaskEditModal received props:', { taskId, taskTitle, taskContent, taskStatus });

  const title = useSignal(taskTitle);
  const content = useSignal(taskContent);
  const status = useSignal<TaskStatus>(taskStatus);

  // Update values when props change or modal opens
  useVisibleTask$(({ track }) => {
    track(() => taskTitle);
    track(() => taskContent);
    track(() => taskStatus);
    track(() => open);

    console.log('Props changed in TaskEditModal:', { taskTitle, taskContent, taskStatus, open });

    if (open) {
      title.value = taskTitle;
      content.value = taskContent;
      status.value = taskStatus;

      // Validate the haiku
      const result = validateHaiku(taskContent);
      validation.isValid = result.isValid;
      validation.syllableCounts = result.syllableCounts;
      validation.feedback = result.feedback;
    }
  });

  const validation = useStore({
    isValid: true, // Assume existing tasks are valid
    syllableCounts: [0, 0, 0],
    feedback: ''
  });

  // Validate haiku as user types
  const validateContent = $(() => {
    const result = validateHaiku(content.value);
    validation.isValid = result.isValid;
    validation.syllableCounts = result.syllableCounts;
    validation.feedback = result.feedback;
  });

  // Generate a random haiku template
  const generateTemplate = $(() => {
    content.value = generateHaikuTemplate();
    validateContent();
  });

  // Submit the task update
  const submitTask = $(() => {
    if (!validation.isValid || !title.value.trim()) {
      return;
    }

    // Update existing task
    updateTodo(taskId, {
      title: title.value,
      content: content.value,
      status: status.value,
      is_completed: status.value === 'done', // For backward compatibility
      updated_at: new Date().toISOString()
    });

    // Close modal
    onClose();
  });

  // Delete task
  const handleDelete = $(() => {
    if (confirm('Are you sure you want to delete this task?')) {
      deleteTodo(taskId);
      onClose();
    }
  });

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Edit Task"
    >
      <div class="space-y-4">
        <div class="space-y-2">
          <Label for="title" class="flex items-center">
            <span class="mr-2">📝</span>
            Task Title
          </Label>
          <Input
            id="title"
            value={title.value}
            onInput$={(e: any) => title.value = e.target.value}
            placeholder="Enter a title for your task"
          />
        </div>

        <div class="space-y-2">
          <div class="flex justify-between items-center">
            <Label for="content" class="flex items-center">
              <span class="mr-2">🪶</span>
              Task Description (as a Haiku)
            </Label>
            <Button
              variant="ghost"
              size="sm"
              onClick$={generateTemplate}
              class="text-xs"
            >
              Generate Template
            </Button>
          </div>
          <Textarea
            id="content"
            value={content.value}
            onInput$={(e: any) => {
              content.value = e.target.value;
              validateContent();
            }}
            class="font-serif h-32"
            placeholder="Write your task as a haiku (5-7-5 syllables)"
          />
        </div>

        <div class="space-y-2">
          <Label for="status" class="flex items-center">
            <span class="mr-2">🏷️</span>
            Status
          </Label>
          <Select
            id="status"
            value={status.value}
            onChange$={(e: any) => status.value = e.target.value}
          >
            {Object.entries(TASK_STATUS_CONFIG).map(([value, config]) => {
              const label = `${config.label} (${config.description})`;
              return (
                <option key={value} value={value}>
                  {label}
                </option>
              );
            })}
          </Select>
        </div>

        <div class="bg-muted/20 p-3 rounded-md">
          <div class="flex justify-between text-xs text-muted-foreground">
            <span>Line 1: {validation.syllableCounts[0] || 0}/5 syllables</span>
            <span>Line 2: {validation.syllableCounts[1] || 0}/7 syllables</span>
            <span>Line 3: {validation.syllableCounts[2] || 0}/5 syllables</span>
          </div>
          <p class={`text-sm mt-1 ${validation.isValid ? 'text-green-500' : 'text-amber-500'}`}>
            {validation.feedback}
          </p>
        </div>
      </div>

      <ModalFooter>
        <Button
          variant="destructive"
          onClick$={handleDelete}
          class="mr-auto"
        >
          Delete
        </Button>
        <Button
          variant="outline"
          onClick$={onClose}
        >
          Cancel
        </Button>
        <Button
          onClick$={submitTask}
          disabled={!validation.isValid || !title.value.trim()}
        >
          Save Changes
        </Button>
      </ModalFooter>
    </Modal>
  );
});
