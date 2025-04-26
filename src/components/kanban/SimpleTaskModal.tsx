import { component$, $, useSignal, useStore, useTask$ } from '@builder.io/qwik';
import { v4 as uuidv4 } from 'uuid';
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
import { addTodo, updateTodo, deleteTodo } from '~/services/yjs-sync';
import { Task, TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';

interface SimpleTaskModalProps {
  isOpen: boolean;
  onClose: () => void;
  editTask: Task | null;
  defaultStatus: TaskStatus;
}

export const SimpleTaskModal = component$<SimpleTaskModalProps>(({
  isOpen,
  onClose,
  editTask,
  defaultStatus
}) => {
  // Form state
  const formTitle = useSignal('');
  const formContent = useSignal('');
  const formStatus = useSignal<TaskStatus>(defaultStatus);
  const isEditing = useSignal(false);

  // Validation state
  const validation = useStore({
    isValid: false,
    syllableCounts: [0, 0, 0],
    prevSyllableCounts: [0, 0, 0],
    feedback: ''
  });

  // Reset form when modal opens/closes or editTask changes
  useTask$(({ track }) => {
    track(() => isOpen);
    track(() => editTask);

    if (isOpen) {
      if (editTask) {
        // Editing mode
        isEditing.value = true;
        formTitle.value = editTask.title || '';
        formContent.value = editTask.content || '';
        formStatus.value = editTask.status || defaultStatus;

        // Validate the haiku
        const result = validateHaiku(editTask.content || '');
        validation.isValid = result.isValid;
        validation.syllableCounts = result.syllableCounts;
        validation.prevSyllableCounts = [...result.syllableCounts];
        validation.feedback = result.feedback;
      } else {
        // Creation mode
        isEditing.value = false;
        formTitle.value = '';
        formContent.value = '';
        formStatus.value = defaultStatus;
        validation.isValid = false;
        validation.syllableCounts = [0, 0, 0];
        validation.prevSyllableCounts = [0, 0, 0];
        validation.feedback = '';
      }
    }
  });

  // Validate haiku as user types
  const validateContent = $(() => {
    const result = validateHaiku(formContent.value);
    validation.isValid = result.isValid;

    // Check if syllable counts have changed
    const syllablesChanged = result.syllableCounts.some(
      (count, index) => count !== validation.prevSyllableCounts[index]
    );

    // Update syllable counts
    validation.syllableCounts = result.syllableCounts;

    // Only update feedback if syllable counts have changed
    if (syllablesChanged) {
      validation.feedback = result.feedback;
      // Store current syllable counts for next comparison
      validation.prevSyllableCounts = [...result.syllableCounts];
    }
  });

  // Generate a random haiku template
  const generateTemplate = $(() => {
    formContent.value = generateHaikuTemplate();
    validateContent();
  });

  // Submit the task
  const submitTask = $(() => {
    if (!validation.isValid || !formTitle.value.trim()) {
      return;
    }

    if (isEditing.value && editTask) {
      // Update existing task
      updateTodo(editTask.id, {
        title: formTitle.value,
        content: formContent.value,
        status: formStatus.value,
        is_completed: formStatus.value === 'done', // For backward compatibility
        updated_at: new Date().toISOString()
      });
    } else {
      // Create new task
      const newTask = {
        id: uuidv4(),
        title: formTitle.value,
        content: formContent.value,
        status: formStatus.value,
        is_completed: formStatus.value === 'done', // For backward compatibility
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        user_id: 'local' // This will be replaced with the actual user ID when synced
      };

      addTodo(newTask);
    }

    // Close modal
    onClose();
  });

  // Delete task
  const handleDelete = $(() => {
    if (editTask && confirm('Are you sure you want to delete this task?')) {
      deleteTodo(editTask.id);
      onClose();
    }
  });

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={isEditing.value ? 'Edit Task' : 'Create New Task'}
    >
      <div class="space-y-4">
        <div class="space-y-2">
          <Label for="title" class="flex items-center">
            <span class="mr-2">📝</span>
            Task Title
          </Label>
          <Input
            id="title"
            value={formTitle.value}
            onInput$={(e: any) => formTitle.value = e.target.value}
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
            value={formContent.value}
            onInput$={(e: any) => {
              formContent.value = e.target.value;
              validateContent();
            }}
            class="font-serif h-32"
            placeholder="Write your task as a haiku (5-7-5 syllables)"
          />
        </div>

        {/* Only show status selector when editing or when creating from global button */}
        {(isEditing.value || !defaultStatus) && (
          <div class="space-y-2">
            <Label for="status" class="flex items-center">
              <span class="mr-2">🏷️</span>
              Status
            </Label>
            <Select
              id="status"
              value={formStatus.value}
              onChange$={(e: any) => formStatus.value = e.target.value}
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
        )}

        <div class="bg-muted/20 p-3 rounded-md">
          <div class="flex justify-between text-xs text-muted-foreground">
            <span class={`px-2 py-1 rounded-l-md transition-all duration-1000 ${validation.syllableCounts[0] === 5 ? 'bg-green-900/10 dark:bg-green-900/20 text-green-800 dark:text-green-400 outline outline-1 outline-green-500/30' : ''}`}>Line 1: {validation.syllableCounts[0] || 0}/5 syllables</span>
            <span class={`px-2 py-1 transition-all duration-1000 ${validation.syllableCounts[1] === 7 ? 'bg-green-900/10 dark:bg-green-900/20 text-green-800 dark:text-green-400 outline outline-1 outline-green-500/30' : ''}`}>Line 2: {validation.syllableCounts[1] || 0}/7 syllables</span>
            <span class={`px-2 py-1 rounded-r-md transition-all duration-1000 ${validation.syllableCounts[2] === 5 ? 'bg-green-900/10 dark:bg-green-900/20 text-green-800 dark:text-green-400 outline outline-1 outline-green-500/30' : ''}`}>Line 3: {validation.syllableCounts[2] || 0}/5 syllables</span>
          </div>
          <p class={`text-sm mt-3 ${validation.isValid ? 'text-green-500' : 'text-amber-500'}`}>
            {validation.feedback}
          </p>
        </div>
      </div>

      <ModalFooter>
        {isEditing.value && (
          <Button
            variant="destructive"
            onClick$={handleDelete}
            class="mr-auto"
          >
            Delete
          </Button>
        )}
        <Button
          variant="outline"
          onClick$={onClose}
        >
          Cancel
        </Button>
        <Button
          onClick$={submitTask}
          disabled={!validation.isValid || !formTitle.value.trim()}
        >
          {isEditing.value ? 'Save Changes' : 'Create Task'}
        </Button>
      </ModalFooter>
    </Modal>
  );
});
