defmodule TodoHaikuWeb.DesignSystem do
  @moduledoc """
  Design system components and utilities following modern UI patterns.
  Provides consistent design tokens, spacing, colors, and animations.
  """
  use Phoenix.Component
  use Gettext, backend: TodoHaikuWeb.Gettext

  # Import verified routes properly
  use Phoenix.VerifiedRoutes, endpoint: TodoHaikuWeb.Endpoint, router: TodoHaikuWeb.Router

  @doc """
  Design tokens for the application.
  """
  def design_tokens do
    %{
      colors: %{
        primary: %{
          50 => "#f0fdf4",
          100 => "#dcfce7",
          500 => "#22c55e",
          600 => "#16a34a",
          700 => "#15803d",
          900 => "#14532d"
        },
        kanban: %{
          blue: %{
            glow: "#5DADE2",
            background: "rgba(93, 173, 226, 0.5)"
          },
          yellow: %{
            glow: "#F5B041",
            background: "rgba(245, 176, 65, 0.5)"
          },
          green: %{
            glow: "#58D68D",
            background: "rgba(88, 214, 141, 0.5)"
          },
          red: %{
            glow: "#EC7063",
            background: "rgba(236, 112, 99, 0.5)"
          }
        }
      },
      spacing: %{
        xs: "0.25rem",
        sm: "0.5rem",
        md: "1rem",
        lg: "1.5rem",
        xl: "2rem",
        "2xl": "3rem"
      },
      radius: %{
        sm: "0.375rem",
        md: "0.5rem",
        lg: "0.75rem",
        xl: "1rem"
      },
      shadows: %{
        sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        md: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
        lg: "0 10px 15px -3px rgb(0 0 0 / 0.1)",
        glow: "0 0 15px var(--glow-color)"
      }
    }
  end

  @doc """
  Renders a card component with consistent styling.

  ## Examples

      <.card>
        <p>Card content</p>
      </.card>

      <.card variant="glass" class="p-6">
        <p>Glassmorphism card</p>
      </.card>
  """
  attr :variant, :string, default: "default", values: ~w(default glass elevated)
  attr :class, :string, default: ""
  attr :rest, :global, include: ~w(id data-* phx-*)
  slot :inner_block, required: true

  def card(assigns) do
    ~H"""
    <div class={card_classes(@variant, @class)} {@rest}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  @doc """
  Renders a button with consistent styling and states.
  """
  attr :variant, :string, default: "primary", values: ~w(primary secondary ghost danger)
  attr :size, :string, default: "md", values: ~w(sm md lg)
  attr :disabled, :boolean, default: false
  attr :loading, :boolean, default: false
  attr :class, :string, default: ""
  attr :rest, :global, include: ~w(type form phx-*)
  slot :inner_block, required: true

  def button(assigns) do
    ~H"""
    <button
      class={button_classes(@variant, @size, @disabled, @loading, @class)}
      disabled={@disabled || @loading}
      {@rest}
    >
      <.loading_spinner :if={@loading} size="sm" />
      <span class={@loading && "opacity-0"}>
        {render_slot(@inner_block)}
      </span>
    </button>
    """
  end

  @doc """
  Renders a loading spinner.
  """
  attr :size, :string, default: "md", values: ~w(sm md lg)
  attr :class, :string, default: ""

  def loading_spinner(assigns) do
    ~H"""
    <div class={spinner_classes(@size, @class)}>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
    </div>
    """
  end

  @doc """
  Renders an icon with consistent sizing.
  """
  attr :name, :string, required: true
  attr :size, :string, default: "md", values: ~w(sm md lg xl)
  attr :class, :string, default: ""
  attr :rest, :global

  def icon(assigns) do
    ~H"""
    <svg
      class={icon_classes(@size, @class)}
      fill="none"
      viewBox="0 0 24 24"
      stroke-width="1.5"
      stroke="currentColor"
      {@rest}
    >
      <.icon_path name={@name} />
    </svg>
    """
  end

  @doc """
  Renders form input with enhanced styling.
  """
  attr :field, Phoenix.HTML.FormField, required: true
  attr :type, :string, default: "text"
  attr :placeholder, :string, default: nil
  attr :class, :string, default: ""
  attr :rest, :global, include: ~w(disabled readonly required)

  def enhanced_input(assigns) do
    ~H"""
    <div class="form-field">
      <input
        type={@type}
        name={@field.name}
        id={@field.id}
        value={Phoenix.HTML.Form.normalize_value(@type, @field.value)}
        placeholder={@placeholder}
        class={input_classes(@class)}
        {@rest}
      />
      <.field_error :for={msg <- @field.errors}>{msg}</.field_error>
    </div>
    """
  end

  @doc """
  Renders field error message.
  """
  attr :class, :string, default: ""
  slot :inner_block, required: true

  def field_error(assigns) do
    ~H"""
    <p class={["field-error", @class]}>
      {render_slot(@inner_block)}
    </p>
    """
  end

  # Helper functions for class generation

  defp card_classes("default", custom_class) do
    [
      "bg-card",
      "text-card-foreground",
      "rounded-xl",
      "border",
      "border-border",
      "shadow-sm",
      custom_class
    ]
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp card_classes("glass", custom_class) do
    [
      "bg-card/45",
      "backdrop-blur-md",
      "rounded-xl",
      "border",
      "border-white/8",
      "shadow-lg",
      custom_class
    ]
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp card_classes("elevated", custom_class) do
    [
      "bg-card",
      "text-card-foreground",
      "rounded-xl",
      "border",
      "border-border",
      "shadow-lg",
      custom_class
    ]
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp button_classes(variant, size, disabled, loading, custom_class) do
    base_classes = [
      "inline-flex",
      "items-center",
      "justify-center",
      "rounded-md",
      "font-medium",
      "transition-colors",
      "focus-visible:outline-none",
      "focus-visible:ring-2",
      "focus-visible:ring-ring",
      "disabled:pointer-events-none",
      "disabled:opacity-50"
    ]

    variant_classes = case variant do
      "primary" -> [
        "bg-primary",
        "text-primary-foreground",
        "hover:bg-primary/90"
      ]
      "secondary" -> [
        "bg-secondary",
        "text-secondary-foreground",
        "hover:bg-secondary/80"
      ]
      "ghost" -> [
        "hover:bg-accent",
        "hover:text-accent-foreground"
      ]
      "danger" -> [
        "bg-destructive",
        "text-destructive-foreground",
        "hover:bg-destructive/90"
      ]
    end

    size_classes = case size do
      "sm" -> ["h-9", "px-3", "text-sm"]
      "md" -> ["h-10", "px-4", "py-2"]
      "lg" -> ["h-11", "px-8", "text-lg"]
    end

    state_classes = []
    state_classes = if disabled, do: ["opacity-50", "cursor-not-allowed" | state_classes], else: state_classes
    state_classes = if loading, do: ["relative" | state_classes], else: state_classes

    (base_classes ++ variant_classes ++ size_classes ++ state_classes ++ [custom_class])
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp spinner_classes(size, custom_class) do
    base_classes = ["spinner"]

    size_classes = case size do
      "sm" -> ["spinner--sm"]
      "md" -> ["spinner--md"]
      "lg" -> ["spinner--lg"]
    end

    (base_classes ++ size_classes ++ [custom_class])
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp icon_classes(size, custom_class) do
    size_classes = case size do
      "sm" -> ["w-4", "h-4"]
      "md" -> ["w-5", "h-5"]
      "lg" -> ["w-6", "h-6"]
      "xl" -> ["w-8", "h-8"]
    end

    (size_classes ++ [custom_class])
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp input_classes(custom_class) do
    base_classes = [
      "flex",
      "h-10",
      "w-full",
      "rounded-md",
      "border",
      "border-input",
      "bg-background",
      "px-3",
      "py-2",
      "text-sm",
      "ring-offset-background",
      "file:border-0",
      "file:bg-transparent",
      "file:text-sm",
      "file:font-medium",
      "placeholder:text-muted-foreground",
      "focus-visible:outline-none",
      "focus-visible:ring-2",
      "focus-visible:ring-ring",
      "focus-visible:ring-offset-2",
      "disabled:cursor-not-allowed",
      "disabled:opacity-50"
    ]

    (base_classes ++ [custom_class])
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end

  defp icon_path(%{name: "hero-plus"} = assigns) do
    ~H"""
    <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
    """
  end

  defp icon_path(%{name: "hero-pencil"} = assigns) do
    ~H"""
    <path stroke-linecap="round" stroke-linejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
    """
  end

  # Add more icon paths as needed
  defp icon_path(%{name: _} = assigns) do
    ~H"""
    <!-- Default icon path -->
    <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
    """
  end
end
