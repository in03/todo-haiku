# JavaScript Libraries for Todo Haiku Kanban Board

To complete the kanban board functionality, you'll need to download and add two JavaScript libraries:

1. **Sortable.js** - For drag and drop functionality
2. **Alpine.js** - For reactive components

## How to Add the Libraries

### Option 1: Download directly from CDN

1. Download Sortable.js from: https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js
2. Download Alpine.js from: https://cdn.jsdelivr.net/npm/alpinejs@3.13.7/dist/cdn.min.js

Save these files to the `assets/vendor` directory with these filenames:
- `sortable.min.js`
- `alpine.min.js`

### Option 2: Use npm

```bash
npm install sortablejs alpinejs --prefix assets
```

Then modify your `app.js` file to use the npm packages instead of the vendor files.

## Implementation Details

The kanban board is already configured in the LiveView template and LiveView module:

1. The `KanbanBoard` hook in `app.js` initializes Sortable.js on the kanban columns
2. When a task is moved, it sends a `task-moved` event to the server
3. The `handle_event("task-moved", ...)` function in the LiveView updates the task status

## Troubleshooting

If drag and drop isn't working:
1. Check browser console for errors
2. Make sure the JavaScript files are properly loaded
3. Verify that the `phx-hook="KanbanBoard"` attribute is on the board element
4. Ensure the column elements have the correct `data-column` attributes

If you need further assistance, consult the documentation:
- Sortable.js: https://github.com/SortableJS/Sortable
- Alpine.js: https://alpinejs.dev/
- Phoenix LiveView Hooks: https://hexdocs.pm/phoenix_live_view/js-interop.html 