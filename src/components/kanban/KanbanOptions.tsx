import { component$, $, Slot, useSignal, useContext } from '@builder.io/qwik';
import { Input, Select, Switch } from '~/components/ui';
import { KanbanContext } from './KanbanContext';

export type SortOption = 'custom' | 'title' | 'date' | 'mood';

export interface KanbanOptionsProps {
  onSearch?: (searchTerm: string) => void;
  onSortChange?: (sortOption: SortOption) => void;
  onQuietModeToggle?: (enabled: boolean) => void;
}

export const KanbanOptions = component$<KanbanOptionsProps>(({
  onSearch,
  onSortChange,
  onQuietModeToggle
}) => {
  const searchTerm = useSignal('');
  const sortOption = useSignal<SortOption>('custom');
  const quietMode = useSignal(false);
  const kanbanContext = useContext(KanbanContext);

  // Handle search input change
  const handleSearchChange = $((e: any) => {
    searchTerm.value = e.target.value;
    onSearch?.(e.target.value);

    // Update context if available
    if (kanbanContext) {
      kanbanContext.searchTerm = e.target.value;
    }
  });

  // Handle sort option change
  const handleSortChange = $((e: any) => {
    sortOption.value = e.target.value as SortOption;
    onSortChange?.(e.target.value as SortOption);

    // Update context if available
    if (kanbanContext) {
      kanbanContext.sortOption = e.target.value as SortOption;
    }
  });

  // Handle quiet mode toggle
  const toggleQuietMode = $(() => {
    quietMode.value = !quietMode.value;
    onQuietModeToggle?.(quietMode.value);

    // Update context if available
    if (kanbanContext) {
      kanbanContext.quietMode = quietMode.value;
    }
  });

  return (
    <div class="mb-6 bg-card border border-border rounded-lg p-4 shadow-sm">
      <div class="flex flex-col md:flex-row gap-4 items-center justify-between">
        <div class="w-full md:w-1/3">
          <Input
            type="text"
            placeholder="Search tasks..."
            value={searchTerm.value}
            onInput$={handleSearchChange}
            class="w-full"
          />
        </div>

        <div class="flex items-center gap-4">
          <div class="flex items-center">
            <label class="mr-2 text-sm whitespace-nowrap">Sort by:</label>
            <Select
              value={sortOption.value}
              onChange$={handleSortChange}
              class="w-32"
            >
              <option value="custom">Custom</option>
              <option value="title">Title</option>
              <option value="date">Date</option>
              <option value="mood">Mood</option>
            </Select>
          </div>

          <div class="flex items-center gap-2">
            <span class="text-sm text-muted-foreground">Quiet:</span>
            <Switch
              checked={quietMode.value}
              onChange$={toggleQuietMode}
              aria-label="Toggle quiet mode"
            />
          </div>
        </div>
      </div>

      <Slot />
    </div>
  );
});
