import { component$, useSignal, useTask$, QwikIntrinsicElements } from '@builder.io/qwik';
import { cn } from '~/lib/utils';

export interface SwitchProps extends Omit<QwikIntrinsicElements['button'], 'onChange$'> {
  checked?: boolean;
  onChange$?: (checked: boolean) => void;
  class?: string;
  disabled?: boolean;
  label?: string;
}

export const Switch = component$<SwitchProps>(({ 
  checked = false, 
  onChange$, 
  class: className, 
  disabled = false,
  label,
  ...props 
}) => {
  const isChecked = useSignal(checked);
  
  // Keep internal state in sync with props
  useTask$(({ track }) => {
    track(() => checked);
    isChecked.value = checked;
  });

  return (
    <button
      type="button"
      role="switch"
      aria-checked={isChecked.value}
      data-state={isChecked.value ? "checked" : "unchecked"}
      disabled={disabled}
      class={cn(
        "peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50",
        isChecked.value ? "bg-primary" : "bg-input",
        className
      )}
      onClick$={() => {
        if (!disabled) {
          const newValue = !isChecked.value;
          isChecked.value = newValue;
          onChange$?.(newValue);
        }
      }}
      {...props}
    >
      <span
        class={cn(
          "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform",
          isChecked.value ? "translate-x-5" : "translate-x-0"
        )}
      />
      {label && (
        <span class="ml-3 text-sm">{label}</span>
      )}
    </button>
  );
});
