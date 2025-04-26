import { component$, Slot, QwikIntrinsicElements } from '@builder.io/qwik';
import { cn } from '~/lib/utils';

export const Select = component$<QwikIntrinsicElements['select'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <select
      class={cn(
        "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    >
      <Slot />
    </select>
  );
});

export const SelectGroup = component$<QwikIntrinsicElements['optgroup'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <optgroup
      class={cn("", className)}
      {...props}
    >
      <Slot />
    </optgroup>
  );
});

export const SelectOption = component$<QwikIntrinsicElements['option'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <option
      class={cn("", className)}
      {...props}
    >
      <Slot />
    </option>
  );
});
