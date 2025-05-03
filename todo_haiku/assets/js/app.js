// If you want to use Phoenix channels, run `mix help phx.gen.channel`
// to get started and then uncomment the line below.
// import "./user_socket.js"

// You can include dependencies in two ways.
//
// The simplest option is to put them in assets/vendor and
// import them using relative paths:
//
//     import "../vendor/some-package.js"
//
// Alternatively, you can `npm install some-package --prefix assets` and import
// them using a path starting with the package name:
//
//     import "some-package"
//

// Include phoenix_html to handle method=PUT/DELETE in forms and buttons.
import "phoenix_html"
// Establish Phoenix Socket and LiveView configuration.
import {Socket} from "phoenix"
import {LiveSocket} from "phoenix_live_view"
import topbar from "../vendor/topbar"

// Load Sortable.js and Alpine.js from CDN
// No need for import statements as we're loading them as scripts

// Define hooks for LiveView 
const Hooks = {}

// Define a hook for the Kanban board using Sortable.js
Hooks.KanbanBoard = {
  mounted() {
    const board = this.el;
    
    // Check if Sortable is available
    if (typeof Sortable === 'undefined') {
      console.error('Sortable.js is not loaded. Please add the script to your HTML.');
      return;
    }
    
    // Initialize Sortable for each column in the board
    const columns = board.querySelectorAll(".kanban-column-content");
    
    columns.forEach(column => {
      new Sortable(column, {
        group: 'tasks',
        animation: 150,
        ghostClass: 'task-ghost',
        chosenClass: 'task-chosen',
        dragClass: 'task-drag',
        onEnd: (evt) => {
          // Get the task id and new status
          const taskId = evt.item.getAttribute('data-task-id');
          const newStatus = evt.to.getAttribute('data-column');
          
          // Calculate the new position in the column
          const newIndex = evt.newIndex;
          
          // Send the update to the server
          this.pushEvent("task-moved", {
            id: taskId,
            status: newStatus,
            position: newIndex
          });
        }
      });
    });
  }
};

// Check if Alpine is available and initialize it
document.addEventListener('DOMContentLoaded', () => {
  if (typeof Alpine !== 'undefined') {
    // Initialize Alpine
    Alpine.start();
  } else {
    console.error('Alpine.js is not loaded. Please add the script to your HTML.');
  }
});

// Configure LiveSocket with hooks and params
let csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content")
let liveSocket = new LiveSocket("/live", Socket, {
  longPollFallbackMs: 2500,
  params: {_csrf_token: csrfToken},
  hooks: Hooks,
  dom: {
    onBeforeElUpdated(from, to) {
      // Check if Alpine is available before using it
      if (typeof Alpine !== 'undefined' && from._x_dataStack) {
        window.Alpine.clone(from, to);
      }
    }
  }
})

// Show progress bar on live navigation and form submits
topbar.config({barColors: {0: "#29d"}, shadowColor: "rgba(0, 0, 0, .3)"})
window.addEventListener("phx:page-loading-start", _info => topbar.show(300))
window.addEventListener("phx:page-loading-stop", _info => topbar.hide())

// connect if there are any LiveViews on the page
liveSocket.connect()

// expose liveSocket on window for web console debug logs and latency simulation:
// >> liveSocket.enableDebug()
// >> liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
// >> liveSocket.disableLatencySim()
window.liveSocket = liveSocket

