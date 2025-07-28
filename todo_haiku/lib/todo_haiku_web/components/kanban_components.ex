defmodule TodoHaikuWeb.KanbanComponents do
  @moduledoc """
  Kanban board components following modern LiveView patterns.
  Provides reusable, battle-tested components for kanban functionality.
  """
  use Phoenix.Component
  use Gettext, backend: TodoHaikuWeb.Gettext

  import Phoenix.HTML, only: [raw: 1, safe_to_string: 1]
  import TodoHaikuWeb.CoreComponents, only: [icon: 1]

  # Import verified routes properly
  use Phoenix.VerifiedRoutes, endpoint: TodoHaikuWeb.Endpoint, router: TodoHaikuWeb.Router

  @doc """
  Renders a complete kanban board with columns and tasks.

  ## Examples

      <.kanban_board tasks={@tasks} search_term={@search_term}>
        <:column status="open" title="Seed" icon="🌱" color="blue" />
        <:column status="doing" title="Tend" icon="🧹" color="yellow" />
        <:column status="done" title="Fruit" icon="🍎" color="green" />
        <:column status="blocked" title="Withheld" icon="🍂" color="red" />
      </.kanban_board>
  """
  attr :tasks, :list, required: true, doc: "List of tasks to display"
  attr :search_term, :string, default: nil, doc: "Search filter term"
  attr :class, :string, default: "", doc: "Additional CSS classes"
  slot :column, required: true do
    attr :status, :string, required: true
    attr :title, :string, required: true
    attr :icon, :string, required: true
    attr :color, :string, required: true
  end

  def kanban_board(assigns) do
    ~H"""
    <div
      id="kanban-board"
      class={["kanban-board", @class]}
      phx-hook="KanbanBoard"
      data-board="true"
    >
      <.kanban_column
        :for={column <- @column}
        status={column.status}
        title={column.title}
        icon={column.icon}
        color={column.color}
        tasks={filter_tasks(@tasks, column.status, @search_term)}
      />
    </div>
    """
  end

  @doc """
  Renders a single kanban column with header and tasks.
  """
  attr :status, :string, required: true
  attr :title, :string, required: true
  attr :icon, :string, required: true
  attr :color, :string, required: true
  attr :tasks, :list, required: true

  def kanban_column(assigns) do
    assigns = assign(assigns, :task_count, length(assigns.tasks))

    ~H"""
    <div
      class={kanban_column_classes(@color, @task_count)}
      data-status={@status}
      data-color={@color}
    >
      <.kanban_column_header
        status={@status}
        title={@title}
        icon={@icon}
        color={@color}
        task_count={@task_count}
      />

      <.kanban_column_content status={@status} tasks={@tasks} />
    </div>
    """
  end

  @doc """
  Renders the column header with title, icon, and action button.
  """
  attr :status, :string, required: true
  attr :title, :string, required: true
  attr :icon, :string, required: true
  attr :color, :string, required: true
  attr :task_count, :integer, required: true

  def kanban_column_header(assigns) do
    ~H"""
    <div class="kanban-column-header" data-color={@color}>
      <div class="header-glow"></div>
      <h3 class="kanban-column-title">
        <span class="column-icon"><%= @icon %></span>
        <%= @title %>
        <.kanban_action_button status={@status} task_count={@task_count} />
      </h3>
    </div>
    """
  end

  @doc """
  Renders the column content area with tasks.
  """
  attr :status, :string, required: true
  attr :tasks, :list, required: true

  def kanban_column_content(assigns) do
    ~H"""
    <div class="kanban-column-content" data-column={@status}>
      <.task_card :for={task <- @tasks} task={task} />

      <.empty_column_message :if={Enum.empty?(@tasks)} />
    </div>
    """
  end

  @doc """
  Renders a task card with proper styling and interactions.
  """
  attr :task, :map, required: true

  def task_card(assigns) do
    ~H"""
    <div
      class="task-card"
      data-task-id={@task.id}
      data-status={@task.status}
    >
      <div class="task-card-header">
        <div class="task-card-content">
          <h4 class="task-title"><%= @task.title %></h4>
        </div>
        <.task_edit_button task={@task} />
      </div>

      <.task_haiku_content content={@task.content} />
    </div>
    """
  end

  @doc """
  Renders the haiku content with proper formatting.
  """
  attr :content, :string, required: true

  def task_haiku_content(assigns) do
    ~H"""
    <div class="haiku-content">
      <%= @content
          |> Phoenix.HTML.html_escape()
          |> safe_to_string()
          |> String.replace("\n", "<br>")
          |> raw() %>
    </div>
    """
  end

  @doc """
  Renders the task edit button (dot indicator).
  """
  attr :task, :map, required: true

  def task_edit_button(assigns) do
    ~H"""
    <.link patch={~p"/tasks/#{@task.id}/edit"} class="task-dot-link">
      <span class="task-dot" data-status={@task.status}></span>
    </.link>
    """
  end

  @doc """
  Renders the column action button (counter/add button).
  """
  attr :status, :string, required: true
  attr :task_count, :integer, required: true

  def kanban_action_button(assigns) do
    ~H"""
    <.link
      patch={~p"/tasks/new?status=#{@status}"}
      class="column-action-button"
      id={"new-task-in-#{@status}"}
    >
      <div class="action-button-content">
        <div class="counter-view">
          <span class="task-counter-number"><%= @task_count %></span>
        </div>
        <div class="plus-view">
          <.icon name="hero-plus" class="w-5 h-5" />
        </div>
      </div>
    </.link>
    """
  end

  @doc """
  Renders empty column message.
  """
  def empty_column_message(assigns) do
    ~H"""
    <div class="empty-column-message">
      Drag tasks here...
    </div>
    """
  end

  # Helper functions

  defp filter_tasks(tasks, status, search_term) do
    tasks
    |> Enum.filter(fn task -> task.status == status end)
    |> filter_by_search(search_term)
  end

  defp filter_by_search(tasks, nil), do: tasks
  defp filter_by_search(tasks, ""), do: tasks
  defp filter_by_search(tasks, search_term) do
    search_lower = String.downcase(search_term)

    Enum.filter(tasks, fn task ->
      String.contains?(String.downcase(task.title), search_lower) ||
      String.contains?(String.downcase(task.content), search_lower)
    end)
  end

  defp kanban_column_classes(color, task_count) do
    base_classes = ["kanban-column"]

    color_class = ["kanban-column--#{color}"]

    count_class = case task_count do
      0 -> ["kanban-column--empty"]
      count when count >= 5 -> ["kanban-column--high"]
      count when count >= 3 -> ["kanban-column--medium"]
      _ -> ["kanban-column--low"]
    end

    Enum.join(base_classes ++ color_class ++ count_class, " ")
  end
end
