import { component$, Slot, QwikIntrinsicElements } from '@builder.io/qwik';
import { cn } from '~/lib/utils';

export const Label = component$<QwikIntrinsicElements['label'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <label
      class={cn(
        "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70",
        className
      )}
      {...props}
    >
      <Slot />
    </label>
  );
});
