export type TaskStatus =
  | 'open'
  | 'doing'
  | 'done'
  | 'blocked';

export interface Task {
  id: string;
  title: string;
  content: string;
  status: TaskStatus;
  created_at: string;
  updated_at: string;
  user_id: string;
  sortOrder?: number;
  isCollapsed?: boolean;
}

export const TASK_STATUS_CONFIG = {
  'open': {
    label: 'Seed',
    description: 'Idea / Backlog',
    haiku: 'The idea breathes, not yet born.'
  },
  'doing': {
    label: 'Tending',
    description: 'In Progress / Doing',
    haiku: 'Hands shape the moment, gently.'
  },
  'done': {
    label: 'Whole',
    description: 'Done',
    haiku: 'The circle closes, quietly full.'
  },
  'blocked': {
    label: 'Withheld',
    description: 'Blocked / On-Hold',
    haiku: 'The world is not ready, nor are you.'
  }
};
