import { component$, Slot, QwikIntrinsicElements } from '@builder.io/qwik';
import { cn } from '~/lib/utils';

export const Card = component$<QwikIntrinsicElements['div'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <div
      class={cn(
        "rounded-lg border border-border bg-card text-card-foreground shadow-sm",
        className
      )}
      {...props}
    >
      <Slot />
    </div>
  );
});

export const CardHeader = component$<QwikIntrinsicElements['div'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <div
      class={cn("flex flex-col space-y-1.5 p-6", className)}
      {...props}
    >
      <Slot />
    </div>
  );
});

export const CardTitle = component$<QwikIntrinsicElements['h3'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <h3
      class={cn("text-2xl font-semibold leading-none tracking-tight", className)}
      {...props}
    >
      <Slot />
    </h3>
  );
});

export const CardDescription = component$<QwikIntrinsicElements['p'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <p
      class={cn("text-sm text-muted-foreground", className)}
      {...props}
    >
      <Slot />
    </p>
  );
});

export const CardContent = component$<QwikIntrinsicElements['div'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <div
      class={cn("p-6 pt-0", className)}
      {...props}
    >
      <Slot />
    </div>
  );
});

export const CardFooter = component$<QwikIntrinsicElements['div'] & { class?: string }>(({ class: className, ...props }) => {
  return (
    <div
      class={cn("flex items-center p-6 pt-0", className)}
      {...props}
    >
      <Slot />
    </div>
  );
});
