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

// Alpine.js for PETAL stack client-side reactivity
import Alpine from 'alpinejs'

// Register Alpine components
Alpine.data('haikuEnforcement', () => ({
  syllableCounts: [0, 0, 0],
  expectedSyllables: [5, 7, 5],
  isShaking: false,
  lastContent: '',
  isProcessingOverLimit: false,

  init() {
    let inputTimer;

    this.handleKeydown = this.handleKeydown.bind(this);

    // Listen for Phoenix events
    window.addEventListener('phx:syllable-update', (e) => {
      if (e.detail && e.detail.syllable_counts) {
        this.updateSyllableCounts(e.detail.syllable_counts);
      }
    });
  },

  updateSyllableCounts(counts) {
    console.log('Updating syllable counts:', counts);
    if (Array.isArray(counts)) {
      this.syllableCounts = counts.length === 3 ? counts : [0, 0, 0];
    } else if (typeof counts === 'string') {
      // Handle comma-separated string from template
      const parsed = counts.split(',').map(n => parseInt(n) || 0);
      this.syllableCounts = parsed.length === 3 ? parsed : [0, 0, 0];
    }
  },

  handleOverLimit() {
    if (this.isProcessingOverLimit) return;
    
    this.isProcessingOverLimit = true;
    console.log('Processing over-limit: sending backspace');
    
    const textarea = document.getElementById('zen-content-textarea');
    if (textarea) {
      // Send a backspace to remove the last character
      const currentPos = textarea.selectionStart;
      if (currentPos > 0) {
        const beforeCursor = textarea.value.substring(0, currentPos - 1);
        const afterCursor = textarea.value.substring(currentPos);
        textarea.value = beforeCursor + afterCursor;
        textarea.setSelectionRange(currentPos - 1, currentPos - 1);
        
        // Trigger input event to notify LiveView
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      // Shake the window
      this.shakeWindow();
    }
    
    // Reset processing flag after a short delay
    setTimeout(() => {
      this.isProcessingOverLimit = false;
    }, 100);
  },

  handleKeydown(event) {
    const textarea = event.target;
    const content = textarea.value;
    const cursorPos = textarea.selectionStart;
    
    console.log('Keydown:', event.key, 'Syllable counts:', this.syllableCounts);

    // Allow special keys
    const allowedKeys = [
      'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
      'Home', 'End', 'PageUp', 'PageDown',
      'Backspace', 'Delete', 'Tab', 'Escape'
    ];

    if (allowedKeys.includes(event.key)) {
      console.log('Allowed navigation key:', event.key);
      return;
    }

    // Allow Ctrl/Cmd combinations
    if (event.ctrlKey || event.metaKey) {
      console.log('Allowed shortcut:', event.key);
      return;
    }

    // Get current line info
    const currentLineIndex = this.getCurrentLineIndex(content, cursorPos);
    console.log('Current line index:', currentLineIndex);
    
    // Only enforce on first 3 lines
    if (currentLineIndex >= 3) {
      console.log('Beyond line 3, no enforcement');
      return;
    }

    const expectedSyllables = this.expectedSyllables[currentLineIndex];
    const currentSyllables = this.syllableCounts[currentLineIndex] || 0;
    
    console.log(`Line ${currentLineIndex}: ${currentSyllables}/${expectedSyllables} syllables`);

    // Check if we're at a word boundary (space or newline before cursor)
    const isAtWordBoundary = cursorPos === 0 || 
                            content[cursorPos - 1] === ' ' || 
                            content[cursorPos - 1] === '\n';

    // Handle SPACE and ENTER at syllable limits
    if ((event.key === ' ' || event.key === 'Enter') && currentSyllables === expectedSyllables && !isAtWordBoundary) {
      console.log('Space/Enter at syllable limit - auto line break');
      event.preventDefault();
      
      if (currentLineIndex < 2) {
        // Insert newline and move to next line
        const beforeCursor = content.substring(0, cursorPos);
        const afterCursor = content.substring(cursorPos);
        const newContent = beforeCursor + '\n' + afterCursor;
        
        textarea.value = newContent;
        textarea.setSelectionRange(cursorPos + 1, cursorPos + 1);
        
        // Trigger input event to notify LiveView
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      } else {
        // At line 3 limit - shake to indicate no more lines
        this.shakeWindow();
      }
      return;
    }

    // Block input if we're already at the syllable limit for this line
    if (currentSyllables >= expectedSyllables && !isAtWordBoundary) {
      console.log('At or over syllable limit, blocking input');
      event.preventDefault();
      this.shakeWindow();
      return;
    }

    // Allow input if under the limit
    console.log('Under syllable limit, allowing input');
  },

  getCurrentLineIndex(content, cursorPos) {
    const beforeCursor = content.substring(0, cursorPos);
    const lineBreaks = (beforeCursor.match(/\n/g) || []).length;
    return Math.min(lineBreaks, 2); // Cap at 2 (third line)
  },

  shakeWindow() {
    console.log('Shaking window');
    this.isShaking = true;
    
    setTimeout(() => {
      this.isShaking = false;
    }, 500);
  }
}))

// Start Alpine
Alpine.start()

// Zen Interface Hooks (simplified)
let ZenHooks = {
  ZenFocus: {
  mounted() {
      this.el.addEventListener('focus', (e) => {
        e.target.classList.add('zen-focus');
        // Add gentle animation on focus
        e.target.style.transform = 'scale(1.02)';
        e.target.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
      });
      
      this.el.addEventListener('blur', (e) => {
        e.target.classList.remove('zen-focus');
        e.target.style.transform = 'scale(1)';
      });
    }
  },

  ZenForm: {
    mounted() {
      // Add zen animations to form elements
      const inputs = this.el.querySelectorAll('input, textarea');
      inputs.forEach(input => {
        input.addEventListener('focus', () => {
          input.style.animation = 'zen-expand 0.4s ease-out';
        });
        
        input.addEventListener('blur', () => {
          input.style.animation = '';
        });
      });
    }
  },

  ZenTransition: {
    mounted() {
      // Add entrance animation
      this.el.style.animation = 'zen-fade-in 0.6s ease-out';
      
      // Add leaf drift animation to leaf icons
      const leafIcons = this.el.querySelectorAll('.zen-leaf-icon');
      leafIcons.forEach(leaf => {
        leaf.style.animation = 'leaf-drift 6s ease-in-out infinite';
      });
    }
  },

  ZenParticles: {
    mounted() {
      // Create subtle particle background effect
      const particles = document.createElement('div');
      particles.className = 'zen-particles';
      document.body.appendChild(particles);
    },
    
    destroyed() {
      const particles = document.querySelector('.zen-particles');
      if (particles) {
        particles.remove();
      }
    }
  },

  ZenFlash: {
    mounted() {
      // Style flash messages for zen interface
      this.el.classList.add('zen-flash');
      
      // Auto-hide after 4 seconds with fade out
      setTimeout(() => {
        this.el.style.animation = 'zen-fade-out 0.5s ease-out forwards';
        setTimeout(() => {
          this.el.remove();
        }, 500);
      }, 4000);
    }
  },

  // Zen Title Input Hook - handles the initial title input and card expansion
  ZenTitleInput: {
    mounted() {
      console.log('ZenTitleInput hook mounted');
      this.el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          const title = this.el.value.trim();
          console.log('Enter pressed, title:', title);
          if (title) {
            // Trigger the start-haiku event with the title
            this.pushEvent('start-haiku', { title: title });
            console.log('Sent start-haiku event');
          }
        }
      });

      // Auto-focus on mount
      setTimeout(() => {
        this.el.focus();
      }, 100);
    }
  },

  // Simplified content input hook - Alpine.js handles enforcement
  ZenContentInput: {
    mounted() {
      console.log('ZenContentInput hook mounted (simplified for Alpine)');
      
      // Auto-focus when the content area appears
      setTimeout(() => {
        this.el.focus();
        this.el.setSelectionRange(this.el.value.length, this.el.value.length);
      }, 700);

      this.el.addEventListener('input', (e) => {
        const content = this.el.value;
        console.log('Input event, content:', content);
        // Trigger validation event for LiveView
        this.pushEvent('validate-content', { content: content });
      });

      // Handle special shortcuts
      this.el.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.pushEvent('cancel-haiku');
        }
        
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
          this.pushEvent('save-haiku');
        }
      });
    },

    updated() {
      // Ensure focus stays on content when re-rendered
      if (document.activeElement !== this.el) {
        this.el.focus();
      }
    }
  },

  // Enhanced Zen Form Hook for overall card management
  ZenCardManager: {
    mounted() {
      console.log('ZenCardManager hook mounted');
      this.cardElement = this.el;
      
      // Add entrance animation
      this.cardElement.style.opacity = '0';
      this.cardElement.style.transform = 'translateY(20px) scale(0.95)';
      
      setTimeout(() => {
        this.cardElement.style.transition = 'all 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55)';
        this.cardElement.style.opacity = '1';
        this.cardElement.style.transform = 'translateY(0) scale(1)';
      }, 50);
    },

    updated() {
      console.log('ZenCardManager updated, checking for expanded state');
      // Handle state transitions
      if (this.cardElement.classList.contains('expanded')) {
        console.log('Card is expanded, setting up content focus');
        // Card expanded - trigger content focus after animation
        setTimeout(() => {
          const contentTextarea = this.cardElement.querySelector('#zen-content-textarea');
          if (contentTextarea) {
            contentTextarea.focus();
            contentTextarea.setSelectionRange(contentTextarea.value.length, contentTextarea.value.length);
          }
        }, 600);
      }
    }
  }
};

// Zen keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // ESC to cancel/return to zen state
  if (e.key === 'Escape') {
    const cancelBtn = document.querySelector('.zen-cancel-btn');
    if (cancelBtn) {
      cancelBtn.click();
    }
  }
  
  // Ctrl/Cmd + N for new haiku
  if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
    e.preventDefault();
    const newBtn = document.querySelector('[phx-click="new-haiku"]');
    if (newBtn) {
      newBtn.click();
    }
  }
  
  // Ctrl/Cmd + L for haiku list
  if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
    e.preventDefault();
    const listBtn = document.querySelector('[phx-click="show-haiku-list"]');
    if (listBtn) {
      listBtn.click();
    }
  }
});

// Zen scroll behavior
let zenScrollTimeout;
window.addEventListener('scroll', () => {
  document.body.classList.add('scrolling');
  clearTimeout(zenScrollTimeout);
  zenScrollTimeout = setTimeout(() => {
    document.body.classList.remove('scrolling');
  }, 150);
});

// Zen mouse movement - subtle UI responses
let zenMouseTimeout;
document.addEventListener('mousemove', () => {
  const controls = document.getElementById('zen-controls');
  if (controls) {
    controls.style.opacity = '1';
    clearTimeout(zenMouseTimeout);
    zenMouseTimeout = setTimeout(() => {
      controls.style.opacity = '0';
    }, 2000);
  }
});

let csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content")
let liveSocket = new LiveSocket("/live", Socket, {
  longPollFallbackMs: 2500,
  params: {_csrf_token: csrfToken},
  hooks: ZenHooks
})

// Show progress bar on live navigation and form submits
topbar.config({barColors: {0: "#29d"}, shadowColor: "rgba(0, 0, 0, .3)"})
window.addEventListener("phx:page-loading-start", _info => topbar.show(300))
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
// << liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
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

// Zen fade-out animation for flash messages
const style = document.createElement('style');
style.textContent = `
  @keyframes zen-fade-out {
    0% {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    100% {
      opacity: 0;
      transform: translateX(-50%) translateY(-20px);
    }
  }
  
  body.scrolling .zen-controls {
    opacity: 0.3 !important;
    transition: opacity 0.2s ease !important;
  }
`;
document.head.appendChild(style);

