import { component$, useSignal, useStore, $ } from '@builder.io/qwik';
import { v4 as uuidv4 } from 'uuid';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
  Button,
  Input,
  Label,
  Textarea,
  Select
} from '~/components/ui';
import { validateHaiku, generateHaikuTemplate } from '~/utils/haiku-validator';
import { addTodo } from '~/services/yjs-sync';
import { TaskStatus, TASK_STATUS_CONFIG } from '~/types/task';

export const TaskForm = component$(() => {
  const content = useSignal('');
  const title = useSignal('');
  const status = useSignal<TaskStatus>('open');
  const isFormOpen = useSignal(false);

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

    const newTask = {
      id: uuidv4(),
      title: title.value,
      content: content.value,
      status: status.value,
      is_completed: status.value === 'done', // For backward compatibility
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      user_id: 'local' // This will be replaced with the actual user ID when synced
    };

    addTodo(newTask);

    // Reset form
    title.value = '';
    content.value = '';
    status.value = 'open';
    validation.isValid = false;
    validation.syllableCounts = [0, 0, 0];
    validation.feedback = '';
    isFormOpen.value = false;
  });

  return (
    <div class="zen-container mb-8">
      {!isFormOpen.value ? (
        <Button
          onClick$={() => isFormOpen.value = true}
          class="w-full"
        >
          Create New Task
        </Button>
      ) : (
        <Card class="border-t-4 border-t-accent">
          <CardHeader>
            <div class="flex items-center mb-2">
              <span class="text-xl mr-2">✨</span>
              <CardTitle>Create a New Task</CardTitle>
            </div>
          </CardHeader>
          <CardContent class="space-y-4">
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
          </CardContent>
          <CardFooter class="flex justify-between border-t border-border pt-4">
            <Button
              variant="outline"
              onClick$={() => isFormOpen.value = false}
            >
              Cancel
            </Button>
            <Button
              onClick$={submitTask}
              disabled={!validation.isValid || !title.value.trim()}
            >
              Create Task
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
});
