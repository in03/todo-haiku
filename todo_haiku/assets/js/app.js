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
    // Update column references to work with both old and new class structures
    const columns = board.querySelectorAll(".kanban-column-content, [data-column]");
    
    columns.forEach(column => {
      new Sortable(column, {
        group: 'tasks',
        animation: 150,
        ghostClass: 'task-ghost',
        chosenClass: 'task-chosen',
        dragClass: 'task-drag',
        handle: '.task-card', 
        draggable: '.task-card', 
        filter: '.column-guide, .text-center, .empty-column-message', 
        forceFallback: true, 
        fallbackClass: 'sortable-fallback',
        scroll: true, 
        scrollSensitivity: 80,
        scrollSpeed: 20,
        
        // When drag starts
        onStart: (evt) => {
          document.body.classList.add('dragging-active');
          
          // Highlight destination columns
          document.querySelectorAll('.kanban-column').forEach(col => {
            if (col !== evt.from.closest('.kanban-column')) {
              col.classList.add('highlight-column');
            }
          });
        },
        
        // When drag ends
        onEnd: (evt) => {
          document.body.classList.remove('dragging-active');
          
          const taskId = evt.item.getAttribute('data-task-id');
          const newStatus = evt.to.getAttribute('data-column') || evt.to.closest('[data-status]')?.getAttribute('data-status');
          const newIndex = evt.newIndex;
          
          // Remove highlight from destination columns
          document.querySelectorAll('.kanban-column').forEach(col => {
            col.classList.remove('highlight-column');
          });
          
          // Send the update to the server
          if (taskId && newStatus) {
            this.pushEvent("task-moved", {
              id: taskId,
              status: newStatus,
              position: newIndex
            });
          }
          
          // Update visual effects
          setTimeout(() => {
            this.updateTaskCardColors();
            this.updateColumnGlowIntensity();
            this.addFloatingEffect();
          }, 300);
        }
      });
    });
    
    // Initialize column effects
    this.setupHoverEffects();
    this.updateTaskCardColors();
    this.updateColumnGlowIntensity();
    this.setupTaskCardTilt();
    // this.setupTaskDotRipple(); // Call removed
    this.setupHeaderGlow();
    
    // Add pulse animation to task counters
    const taskCounters = board.querySelectorAll('.task-counter');
    taskCounters.forEach(counter => {
      // counter.style.transition = 'all 0.3s ease'; // CSS now handles specific transitions
      counter.addEventListener('mouseenter', () => {
        counter.style.transform = 'translateY(-50%) scale(1.2)'; // Combine transforms
        counter.style.boxShadow = `0 0 12px var(--glow-color)`;
        counter.style.textShadow = `0 0 10px var(--glow-color)`;
      });
      counter.addEventListener('mouseleave', () => {
        counter.style.transform = 'translateY(-50%) scale(1)'; // Combine transforms and reset scale
        counter.style.boxShadow = `0 0 8px var(--glow-color)`;
        counter.style.textShadow = `0 0 8px var(--glow-color)`;
      });
    });
  },
  
  setupHoverEffects() {
    // Add hover effects to columns
    const columns = this.el.querySelectorAll('.kanban-column');
    columns.forEach(column => {
      const header = column.querySelector('.kanban-column-header');
      
      // header.addEventListener('mouseenter', () => { // Keep for glow, remove transform
        // header.style.transform = 'translateY(-3px)';
        // header.style.boxShadow = `0 8px 24px rgba(0, 0, 0, 0.3), 0 0 calc(var(--glow-intensity, 5px) * 1.5) var(--glow-color)`;
      // });
      
      // header.addEventListener('mouseleave', () => { // Keep for glow, remove transform
        // header.style.transform = 'translateY(0)';
        // header.style.boxShadow = `0 4px 12px rgba(0, 0, 0, 0.2), 0 0 var(--glow-intensity, 5px) var(--glow-color)`;
      // });
    });
    
    // Add hover effects to task cards for extra glow
    const cards = this.el.querySelectorAll('.task-card');
    cards.forEach(card => {
      card.style.animation = 'none';
      card.addEventListener('mouseenter', () => {
        // 3D tilt and glow
        card.classList.add('task-card-tilt');
        card.style.transition = 'box-shadow 0.18s cubic-bezier(.4,0,.2,1), transform 0.18s cubic-bezier(.4,0,.2,1)';
        card.style.boxShadow = `0 8px 32px 0 rgba(0,0,0,0.28), 0 0 24px 4px var(--glow-color, #9d8cff)`;
      });
      card.addEventListener('mousemove', (e) => {
        // Subtle 3D tilt
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = ((y - centerY) / centerY) * 6;
        const rotateY = ((x - centerX) / centerX) * -6;
        card.style.transform = `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.025)`;
      });
      card.addEventListener('mouseleave', () => {
        card.classList.remove('task-card-tilt');
        card.style.transform = 'none';
        card.style.boxShadow = 'none';
        card.style.zIndex = '1';
      });
    });
  },
  
  updateTaskCardColors() {
    // Update task cards to inherit column color variables
    const columns = this.el.querySelectorAll('.kanban-column');
    columns.forEach(column => {
      const glowColor = getComputedStyle(column).getPropertyValue('--glow-color');
      const taskCards = column.querySelectorAll('.task-card');
      const guides = column.querySelectorAll('.column-guide');
      
      taskCards.forEach(card => {
        // Ensure task cards show correct color styling from their column
        card.style.setProperty('--glow-color', glowColor);
      });
      
      // Also set the guide colors
      guides.forEach(guide => {
        guide.style.background = `linear-gradient(90deg, transparent, ${glowColor}, transparent)`;
      });
    });
  },
  
  updateColumnGlowIntensity() {
    // Update column glow intensity based on task count
    const columns = this.el.querySelectorAll('.kanban-column');
    columns.forEach(column => {
      const taskCount = column.querySelectorAll('.task-card').length;
      const counter = column.querySelector('.task-counter');
      const header = column.querySelector('.kanban-column-header');
      
      // Apply appropriate glow class based on task count
      column.classList.remove('task-count-low', 'task-count-medium', 'task-count-high');
      
      if (taskCount >= 5) {
        column.classList.add('task-count-high');
      } else if (taskCount >= 3) {
        column.classList.add('task-count-medium');
      } else if (taskCount > 0) {
        column.classList.add('task-count-low');
      }
      
      // Apply dimming for empty columns
      if (taskCount === 0) {
        column.classList.add('kanban-column-empty');
      } else {
        column.classList.remove('kanban-column-empty');
      }
      
      // Update counter if it exists
      if (counter) {
        counter.textContent = taskCount;
      }
    });
  },
  
  addFloatingEffect() {
    // Add subtle floating animation to cards for more life
    const cards = this.el.querySelectorAll('.task-card');
    cards.forEach((card, index) => {
      // Create different timings for each card
      const animationDuration = 3 + (index % 3); // 3-5 seconds
      const animationDelay = index * 0.2; // Stagger the animations
      
      card.style.animation = `float ${animationDuration}s ease-in-out ${animationDelay}s infinite alternate`;
    });
  },
  
  setupTaskCardTilt() {
    // Already handled in setupHoverEffects
  },
  
  setupTaskDotRipple() {
    // const dots = this.el.querySelectorAll('.task-dot');
    // dots.forEach(dot => {
    //   dot.addEventListener('click', (e) => {
    //     const card = dot.closest('.task-card');
    //     if (!card) return;
    //     const existingOverlay = card.querySelector('.dot-expand-overlay');
    //     if (existingOverlay) existingOverlay.remove();
    //     const overlay = document.createElement('span');
    //     overlay.className = 'dot-expand-overlay';
    //     const cardRect = card.getBoundingClientRect();
    //     const dotRect = dot.getBoundingClientRect();
    //     const initialX = dotRect.left - cardRect.left + dotRect.width / 2;
    //     const initialY = dotRect.top - cardRect.top + dotRect.height / 2;
    //     overlay.style.left = `${initialX}px`;
    //     overlay.style.top = `${initialY}px`;
    //     overlay.style.width = '0';
    //     overlay.style.height = '0';
    //     overlay.style.borderRadius = '50%';
    //     overlay.style.position = 'absolute';
    //     overlay.style.zIndex = '0';
    //     overlay.style.backgroundColor = getComputedStyle(dot).getPropertyValue('background-color');
    //     overlay.style.pointerEvents = 'none';
    //     overlay.style.transform = 'translate(-50%, -50%) scale(0)';
    //     overlay.style.opacity = '0'; 
    //     const originalOverflow = card.style.overflow;
    //     card.style.overflow = 'hidden';
    //     card.appendChild(overlay);
    //     const dx = Math.max(initialX, cardRect.width - initialX);
    //     const dy = Math.max(initialY, cardRect.height - initialY);
    //     const rippleRadius = Math.sqrt(dx * dx + dy * dy);
    //     const initialSize = Math.max(dotRect.width, dotRect.height, 1);
    //     const scaleFactor = (rippleRadius * 2) / initialSize; 
    //     requestAnimationFrame(() => {
    //       overlay.style.transition = `transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.25s cubic-bezier(0.4, 0, 0.2, 1)`;
    //       overlay.style.transform = `translate(-50%, -50%) scale(${scaleFactor * 1.1})`;
    //       overlay.style.opacity = '0.3';
    //     });
    //     setTimeout(() => {
    //       overlay.style.transition = `opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1)`;
    //       overlay.style.opacity = '0';
    //     }, 250); 
    //     setTimeout(() => {
    //       overlay.remove();
    //       card.style.overflow = originalOverflow;
    //     }, 450); 
    //   });
    // });
    // Functionality removed for now.
  },
  
  setupHeaderGlow() {
    // Mouse-following glow for column headers
    const headers = this.el.querySelectorAll('.kanban-column-header');
    headers.forEach(header => {
      const glow = header.querySelector('.header-glow');
      header.addEventListener('mousemove', (e) => {
        const rect = header.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        glow.style.display = 'block';
        header.classList.add('glow-active');
        glow.style.background = `radial-gradient(circle at ${x}% ${y}%, var(--glow-color, #9d8cff) 0%, transparent 70%)`;
      });
      header.addEventListener('mouseleave', () => {
        header.classList.remove('glow-active');
        glow.style.display = 'none';
      });
    });
  }
};

// Alpine.js initialization handling
document.addEventListener('DOMContentLoaded', () => {
  // Only initialize if manual initialization is enabled and Alpine isn't already initialized
  if (window._x_alpineInitManual && typeof Alpine !== 'undefined' && !window.Alpine?.initialized) {
    // Check if Alpine is already initialized (Alpine internally sets this up)
    if (!document.querySelector('[x-data]')?.hasAttribute('data-alpine-initialized')) {
      // Set a flag to prevent double initialization
      window.Alpine.initialized = true;
      console.log('Initializing Alpine.js from app.js');
      // Start Alpine
      Alpine.start();
    } else {
      console.log('Alpine.js already initialized, skipping manual initialization');
    }
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
      // Check if Alpine is available and properly initialized
      if (typeof Alpine !== 'undefined' && window.Alpine?.initialized) {
        if (from._x_dataStack) {
          Alpine.clone(from, to);
        }
      }
    }
  }
})

// Show progress bar on live navigation and form submits
topbar.config({barColors: {0: "#29d"}, shadowColor: "rgba(0, 0, 0, .3)"})
window.addEventListener("phx:page-loading-start", _info => topbar.show())
window.addEventListener("phx:page-loading-stop", _info => topbar.hide())

// Handle delete links with CSRF tokens
document.addEventListener("DOMContentLoaded", () => {
  document.addEventListener("click", (e) => {
    // Find links with data-method="delete"
    const element = e.target.closest("a[data-method='delete']");
    if (element) {
      e.preventDefault();
      
      // Create a hidden form to submit the delete request
      const form = document.createElement("form");
      form.method = "post";
      form.action = element.href;
      form.style.display = "none";
      
      // Add method override
      const methodInput = document.createElement("input");
      methodInput.setAttribute("type", "hidden");
      methodInput.setAttribute("name", "_method");
      methodInput.setAttribute("value", "delete");
      form.appendChild(methodInput);
      
      // Add CSRF token
      const csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content");
      const csrfInput = document.createElement("input");
      csrfInput.setAttribute("type", "hidden");
      csrfInput.setAttribute("name", "_csrf_token");
      csrfInput.setAttribute("value", csrfToken);
      form.appendChild(csrfInput);
      
      // Append to body and submit
      document.body.appendChild(form);
      form.submit();
    }
  });
});

// connect if there are any LiveViews on the page
liveSocket.connect()

// expose liveSocket on window for web console debug logs and latency simulation:
// >> liveSocket.enableDebug()
// >> liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
// >> liveSocket.disableLatencySim()
window.liveSocket = liveSocket

// Add shortcut for search focus (Cmd+K / Ctrl+K)
window.addEventListener('keydown', (event) => {
  const searchInput = document.getElementById('global-search-input');
  if (!searchInput) return; // Only run if search input exists

  if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
    event.preventDefault();
    searchInput.focus();
  }
});

