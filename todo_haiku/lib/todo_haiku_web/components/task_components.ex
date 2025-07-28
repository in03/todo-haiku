defmodule TodoHaikuWeb.TaskComponents do
  @moduledoc """
  Task-specific UI components for the Todo Haiku application.
  """
  use Phoenix.Component
  use Gettext, backend: TodoHaikuWeb.Gettext

  import TodoHaikuWeb.CoreComponents

  # Import verified routes properly
  use Phoenix.VerifiedRoutes, endpoint: TodoHaikuWeb.Endpoint, router: TodoHaikuWeb.Router

  alias TodoHaikuWeb.DesignSystem

  @doc """
  Renders an empty state when no tasks exist.
  """
  def empty_state(assigns) do
    ~H"""
    <DesignSystem.card variant="glass" class="text-center py-10">
      <div class="mb-4">
        <span class="text-6xl">🍃</span>
      </div>
      <h3 class="text-xl font-semibold mb-2">No haiku tasks yet</h3>
      <p class="text-muted-foreground mb-6">Create your first poetic task to get started</p>
      <DesignSystem.button>
        <.link patch={~p"/tasks/new"}>
          Create First Task
        </.link>
      </DesignSystem.button>
    </DesignSystem.card>
    """
  end

  @doc """
  Renders the complete task form with haiku validation.
  """
  attr :form, Phoenix.HTML.Form, required: true
  attr :page_title, :string, required: true
  attr :validation_state, :map, required: true
  attr :trigger_submit, :boolean, default: false

  def task_form(assigns) do
    ~H"""
    <DesignSystem.card variant="glass" class="max-w-xl mx-auto my-12 p-8">
      <div class="mb-6">
        <h2 class="text-2xl font-bold mb-1"><%= @page_title %></h2>
        <p class="text-muted-foreground text-sm">Create a new task with a haiku description</p>
      </div>

      <.simple_form
        for={@form}
        id="task-form"
        phx-change="validate"
        phx-submit="save"
        phx-trigger-action={@trigger_submit}
      >
        <.input
          field={@form[:title]}
          type="text"
          placeholder="Task Title"
          class="bg-muted/80 text-foreground border border-border rounded-md focus:ring-primary"
        />

        <.haiku_input_section
          form={@form}
          validation_state={@validation_state}
        />

        <.input
          field={@form[:status]}
          type="select"
          options={status_options()}
          label="Status"
          class="bg-muted/80 text-foreground border border-border rounded-md focus:ring-primary"
        />

        <:actions>
          <.validation_debug_info validation_state={@validation_state} />

          <DesignSystem.button
            type="submit"
            disabled={!@validation_state.is_valid || @form[:title].value == ""}
            loading={@trigger_submit}
            class="w-full"
          >
            <%= if @validation_state.is_valid && @form[:title].value != "", do: "Send 🫴", else: "Complete haiku" %>
          </DesignSystem.button>
        </:actions>
      </.simple_form>
    </DesignSystem.card>
    """
  end

  @doc """
  Renders the haiku input section with syllable validation.
  """
  attr :form, Phoenix.HTML.Form, required: true
  attr :validation_state, :map, required: true

  def haiku_input_section(assigns) do
    ~H"""
    <div class="mb-4">
      <div class="flex justify-between items-center mb-1">
        <span class="text-sm text-foreground">Wax poetic...</span>
        <button
          type="button"
          phx-click="generate_template"
          class="text-xs text-primary hover:text-primary/80 transition-colors"
        >
          ✨ Example
        </button>
      </div>

      <.input
        field={@form[:content]}
        type="textarea"
        rows="3"
        phx-debounce="300"
        placeholder="Write your haiku here..."
        class="font-serif resize-none bg-muted/80 text-foreground border border-border rounded-md focus:ring-primary"
        required
      />

      <.syllable_counter_display validation_state={@validation_state} />

      <.haiku_feedback :if={should_show_feedback(@form[:content], @validation_state)} />
    </div>
    """
  end

  @doc """
  Renders the syllable counter display for haiku validation.
  """
  attr :validation_state, :map, required: true

  def syllable_counter_display(assigns) do
    ~H"""
    <div class="mt-3 flex justify-between text-xs text-muted-foreground">
      <.syllable_line_counter
        line_number={1}
        expected={5}
        actual={Enum.at(@validation_state.syllable_counts, 0) || 0}
        position="left"
      />
      <.syllable_line_counter
        line_number={2}
        expected={7}
        actual={Enum.at(@validation_state.syllable_counts, 1) || 0}
        position="center"
      />
      <.syllable_line_counter
        line_number={3}
        expected={5}
        actual={Enum.at(@validation_state.syllable_counts, 2) || 0}
        position="right"
      />
    </div>
    """
  end

  @doc """
  Renders an individual syllable line counter.
  """
  attr :line_number, :integer, required: true
  attr :expected, :integer, required: true
  attr :actual, :integer, required: true
  attr :position, :string, required: true, values: ~w(left center right)

  def syllable_line_counter(assigns) do
    assigns = assign(assigns, :is_correct, assigns.actual == assigns.expected)

    ~H"""
    <div class={["flex-1 relative z-10", position_margin_class(@position)]}>
      <span class={[
        "block py-1 px-2 z-10 relative transition-colors duration-300",
        @is_correct && "text-foreground",
        !@is_correct && "text-muted-foreground"
      ]}>
        Line <%= @line_number %>: <%= @actual %>/<%= @expected %>
      </span>
      <div class={[
        "absolute top-0 left-0 right-0 bottom-0 transition-all duration-700 ease-in-out",
        border_radius_class(@position),
        @is_correct && "bg-primary/10 border border-primary",
        !@is_correct && "bg-transparent"
      ]}></div>
    </div>
    """
  end

  @doc """
  Renders haiku validation feedback.
  """
  def haiku_feedback(assigns) do
    ~H"""
    <p class="text-sm mt-2 text-yellow-500">
      Please enter a <a href="https://en.wikipedia.org/wiki/Haiku_in_English#Syllables" target="_blank" class="underline hover:text-yellow-400">valid haiku</a> (5-7-5 syllables)
    </p>
    """
  end

  @doc """
  Renders validation debug information.
  """
  attr :validation_state, :map, required: true

  def validation_debug_info(assigns) do
    ~H"""
    <div class="mb-2 p-2 bg-muted/70 rounded text-xs text-muted-foreground">
      <p>Haiku valid: <%= @validation_state.is_valid %> | Line syllables: <%= inspect(@validation_state.syllable_counts || [0,0,0]) %></p>
    </div>
    """
  end

  # Private helper functions

  defp should_show_feedback(content_field, validation_state) do
    !validation_state.is_valid &&
    content_field.value &&
    content_field.value != ""
  end

  defp status_options do
    [
      {"Seed 🌱", "open"},
      {"Tend 🧹", "doing"},
      {"Fruit 🍎", "done"},
      {"Withheld 🍂", "blocked"}
    ]
  end

  defp position_margin_class("center"), do: "-mx-[1px]"
  defp position_margin_class("right"), do: "-ml-[1px]"
  defp position_margin_class("left"), do: ""

  defp border_radius_class("left"), do: "rounded-l-md"
  defp border_radius_class("right"), do: "rounded-r-md"
  defp border_radius_class("center"), do: ""
end
