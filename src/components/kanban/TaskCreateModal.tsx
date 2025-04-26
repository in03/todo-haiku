import { component$, $, useSignal, useStore } from '@builder.io/qwik';
import { v4 as uuidv4 } from 'uuid';
import {
  Button,
  Input,
  Label,
  Textarea,
  Modal,
  ModalFooter
} from '~/components/ui';
import { validateHaiku, generateHaikuTemplate } from '~/utils/haiku-validator';
import { addTodo } from '~/services/yjs-sync';
import { TaskStatus } from '~/types/task';

interface TaskCreateModalProps {
  open: boolean;
  onClose: () => void;
  defaultStatus: TaskStatus;
}

export const TaskCreateModal = component$<TaskCreateModalProps>(({ 
  open, 
  onClose, 
  defaultStatus 
}) => {
  const title = useSignal('');
  const content = useSignal('');
  
  const validation = useStore({
    isValid: false,
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

  // Submit the task
  const submitTask = $(() => {
    if (!validation.isValid || !title.value.trim()) {
      return;
    }

    // Create new task
    const newTask = {
      id: uuidv4(),
      title: title.value,
      content: content.value,
      status: defaultStatus,
      is_completed: defaultStatus === 'done', // For backward compatibility
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      user_id: 'local' // This will be replaced with the actual user ID when synced
    };

    addTodo(newTask);

    // Reset form
    title.value = '';
    content.value = '';
    validation.isValid = false;
    validation.syllableCounts = [0, 0, 0];
    validation.feedback = '';

    // Close modal
    onClose();
  });

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Create New Task"
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
          variant="outline" 
          onClick$={onClose}
        >
          Cancel
        </Button>
        <Button
          onClick$={submitTask}
          disabled={!validation.isValid || !title.value.trim()}
        >
          Create Task
        </Button>
      </ModalFooter>
    </Modal>
  );
});
