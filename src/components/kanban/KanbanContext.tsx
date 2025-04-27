import { createContextId } from '@builder.io/qwik';
import { SortOption } from './KanbanOptions';

export interface KanbanContextState {
  searchTerm: string;
  sortOption: SortOption;
  quietMode: boolean;
}

export const KanbanContext = createContextId<KanbanContextState>('kanban-context');
