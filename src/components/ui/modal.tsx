import { component$, Slot, useSignal, useVisibleTask$, $, useOnDocument, QwikIntrinsicElements } from '@builder.io/qwik';
import { cn } from '~/lib/utils';

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  description?: string;
  class?: string;
}

export const Modal = component$<ModalProps>(({ open, onClose, title, description, class: className }) => {
  const modalRef = useSignal<HTMLDivElement | null>(null);
  
  // Close on escape key
  useOnDocument('keydown', $((event: KeyboardEvent) => {
    if (event.key === 'Escape' && open) {
      onClose();
    }
  }));
  
  // Prevent body scroll when modal is open
  useVisibleTask$(({ track }) => {
    track(() => open);
    
    if (open) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
    };
  });
  
  // Handle click outside to close
  const handleBackdropClick = $((e: MouseEvent) => {
    if (modalRef.value && !modalRef.value.contains(e.target as Node)) {
      onClose();
    }
  });
  
  if (!open) return null;
  
  return (
    <div 
      class="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick$={handleBackdropClick}
    >
      <div 
        ref={modalRef} 
        class={cn(
          "bg-card border border-border rounded-lg shadow-lg w-full max-w-md max-h-[90vh] overflow-auto",
          className
        )}
      >
        {title && (
          <div class="p-6 border-b border-border">
            <h2 class="text-xl font-semibold">{title}</h2>
            {description && <p class="text-sm text-muted-foreground mt-1">{description}</p>}
          </div>
        )}
        
        <div class="p-6">
          <Slot />
        </div>
      </div>
    </div>
  );
});

export const ModalFooter = component$<QwikIntrinsicElements['div'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <div 
      class={cn("flex justify-end space-x-2 pt-4 border-t border-border mt-4", className)}
      {...props}
    >
      <Slot />
    </div>
  );
});
