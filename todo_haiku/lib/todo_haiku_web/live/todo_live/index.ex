defmodule TodoHaikuWeb.TodoLive.Index do
  use TodoHaikuWeb, :live_view

  alias TodoHaiku.Todos
  alias TodoHaiku.Todos.Task
  alias TodoHaiku.HaikuValidator

  @impl true
  def mount(_params, _session, socket) do
    tasks = Todos.list_tasks()

    {:ok,
     socket
     |> assign(:tasks, tasks)
     |> assign(:filter, "all")
     |> assign(:search_term, nil)
     |> assign(:task, nil)
     |> assign(:debug_info, %{
         last_validation: nil,
         validation_count: 0,
         is_valid: false,
         syllable_counts: [0, 0, 0],
         over_limit_line: nil
       })
     |> assign(:page_title, "TodoHaiku")}
  end

  @impl true
  def handle_params(params, _url, socket) do
    {:noreply, apply_action(socket, socket.assigns.live_action, params)}
  end

  defp apply_action(socket, :index, _params) do
    socket
    |> assign(:page_title, "TodoHaiku")
    |> assign(:task, nil)
  end

  defp apply_action(socket, :list, _params) do
    socket
    |> assign(:page_title, "All Haikus")
    |> assign(:task, nil)
    |> assign(:form, nil)
  end

  defp apply_action(socket, :new, params) do
    # Get status from query params, default to "open" if missing
    status = Map.get(params, "status", "open")

    # Initialize with empty template for better UX
    task = %Task{
      is_valid_haiku: false,
      syllable_counts: [0, 0, 0],
      feedback: "Enter your haiku",
      status: status # Pre-fill status
    }

    changeset = Todos.change_task(task)
    IO.puts("Initial changeset for new task: #{inspect(changeset.changes)}")
    IO.puts("Initial task data: #{inspect(task)}")

    socket
    |> assign(:page_title, "New Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(changeset))
  end

  defp apply_action(socket, :edit, %{"id" => id}) do
    task = Todos.get_task!(id)

    # Get the content and validate it for the form
    content = task.content
    {is_valid, syllable_counts, feedback} =
      if content == "" do
        {false, [0, 0, 0], "A haiku is required."}
      else
        TodoHaiku.HaikuValidator.validate_haiku(content)
      end

    # Create changeset with validation
    changeset = Todos.change_task(task)

    # Add validation results
    changeset =
      changeset
      |> Ecto.Changeset.put_change(:is_valid_haiku, is_valid)
      |> Ecto.Changeset.put_change(:syllable_counts, syllable_counts)
      |> Ecto.Changeset.put_change(:feedback, feedback)

    # Update debug info with validation results
    debug_info = %{
      last_validation: DateTime.utc_now(),
      validation_count: 1,
      is_valid: is_valid,
      syllable_counts: syllable_counts
    }

    socket
    |> assign(:page_title, "Edit Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(changeset))
    |> assign(:debug_info, debug_info)
  end

  # Generate template haiku - always fails validation to encourage custom content
  defp generate_template() do
    templates = [
      "Here in this moment\nI ponder what words to write\nMy heart speaks softly",
      "Morning light breaking\nThrough windows of my spirit\nNew day calls to me",
      "Gentle rain falling\nWashing away yesterday\nFresh starts everywhere"
    ]
    Enum.random(templates)
  end

  # Handle any other messages (async handlers removed since we use direct inference)
  @impl true
  def handle_info(_message, socket) do
    {:noreply, socket}
  end

  # Event handlers grouped together
  @impl true
  def handle_event("search", %{"value" => search_term}, socket) do
    tasks = if search_term != "" do
      # Simple search by title and content
      all_tasks = Todos.list_tasks()
      search_lower = String.downcase(search_term)

      Enum.filter(all_tasks, fn task ->
        title_match = task.title && String.contains?(String.downcase(task.title), search_lower)
        content_match = task.content && String.contains?(String.downcase(task.content), search_lower)
        title_match || content_match
      end)
    else
      Todos.list_tasks()
    end
    {:noreply, assign(socket, tasks: tasks, search_term: search_term)}
  end

  @impl true
  def handle_event("search", params, socket) do
    IO.inspect(params, label: "Unexpected search params")
    {:noreply, socket}
  end

  @impl true
  def handle_event("filter", %{"filter" => filter}, socket) do
    tasks = case filter do
      "all" -> Todos.list_tasks()
      status -> Todos.list_tasks_by_status(status)
    end
    {:noreply, assign(socket, tasks: tasks, filter: filter)}
  end

  @impl true
  def handle_event("delete", %{"id" => id}, socket) do
    task = Todos.get_task!(id)
    {:ok, _} = Todos.delete_task(task)
    tasks = Todos.list_tasks()
    {:noreply, assign(socket, tasks: tasks)}
  end

  @impl true
  def handle_event("start-haiku", %{"title" => title}, socket) do
    # Create new task with the provided title
    task = %Task{
      title: title,
      content: "",
      is_valid_haiku: false,
      syllable_counts: [0, 0, 0],
      feedback: "Enter your haiku",
      status: "open"
    }

    {:noreply,
     socket
     |> assign(:task, task)
     |> assign(:page_title, "New Haiku")}
  end

  @impl true
  def handle_event("validate-content", %{"content" => content}, socket) do
    task = socket.assigns.task
    updated_task = %{task | content: content}

    # Get previous syllable counts for over-limit detection
    previous_syllable_counts = socket.assigns.debug_info.syllable_counts || [0, 0, 0]

    # Validate the haiku content
    validation_result = HaikuValidator.validate_haiku(content)

    # Handle validation result - HaikuValidator returns {is_valid, syllable_counts, feedback}
    {is_valid, syllable_counts, feedback} = case validation_result do
      {valid, counts, msg} when is_boolean(valid) and is_list(counts) and is_binary(msg) ->
        {valid, ensure_three_counts(counts), msg}
      _ ->
        {false, [0, 0, 0], "Invalid validation result"}
    end

    updated_task = %{updated_task |
      syllable_counts: syllable_counts,
      is_valid_haiku: is_valid,
      feedback: feedback
    }

    # Check for over-limit situation (went from at limit to over limit)
    over_limit_line = detect_over_limit(previous_syllable_counts, syllable_counts)
    IO.puts("Over-limit detection: previous=#{inspect(previous_syllable_counts)}, current=#{inspect(syllable_counts)}, result=#{inspect(over_limit_line)}")

    # Update debug info with all necessary fields
    debug_info = %{
      syllable_counts: syllable_counts,
      is_valid: is_valid,
      last_validation: DateTime.utc_now(),
      validation_count: socket.assigns.debug_info.validation_count + 1,
      over_limit_line: over_limit_line
    }

    socket_with_updates = socket
      |> assign(:task, updated_task)
      |> assign(:debug_info, debug_info)
      |> push_event("syllable-update", %{
        syllable_counts: syllable_counts,
        over_limit_line: over_limit_line
      })

    {:noreply, socket_with_updates}
  end

  @impl true
  def handle_event("cancel-haiku", _params, socket) do
    {:noreply,
     socket
     |> assign(:task, nil)
     |> assign(:page_title, "TodoHaiku")}
  end

  @impl true
  def handle_event("save-haiku", _params, socket) do
    task = socket.assigns.task

    if task && task.is_valid_haiku && task.title && String.trim(task.title) != "" do
      case Todos.create_task(%{
        title: task.title,
        content: task.content,
        status: "open"
      }) do
        {:ok, _created_task} ->
          {:noreply,
           socket
           |> assign(:task, nil)
           |> assign(:tasks, Todos.list_tasks())
           |> assign(:page_title, "TodoHaiku")
           |> put_flash(:info, "Haiku saved!")}

        {:error, _changeset} ->
          {:noreply,
           socket
           |> put_flash(:error, "Could not save haiku")}
      end
    else
      {:noreply, socket}
    end
  end

  @impl true
  def handle_event("generate_template", _, socket) do
    template_content = generate_template()
    changeset = Todos.change_task(socket.assigns.template_task, %{content: template_content})
    {:noreply, assign(socket, changeset: changeset)}
  end

  @impl true
  def handle_event("task-moved", %{"id" => id, "status" => new_status, "position" => position}, socket) do
    task = Todos.get_task!(id)
    {:ok, _task} = Todos.update_task(task, %{status: new_status, position: position})
    tasks = Todos.list_tasks()
    {:noreply, assign(socket, tasks: tasks)}
  end

  @impl true
  def handle_event("show-haiku-list", _params, socket) do
    {:noreply,
     socket
     |> assign(:live_action, :list)
     |> assign(:page_title, "All Haikus")}
  end

  @impl true
  def handle_event("close-list", _params, socket) do
    {:noreply,
     socket
     |> assign(:live_action, :index)
     |> assign(:page_title, "TodoHaiku")}
  end

  @impl true
  def handle_event("edit-haiku", %{"id" => id}, socket) do
    task = Todos.get_task!(id)

    # Validate the existing haiku content for editing
    {is_valid, syllable_counts} = case task.content do
      nil -> {false, [0, 0, 0]}
      "" -> {false, [0, 0, 0]}
      content ->
        case HaikuValidator.validate_haiku(content) do
          {valid, counts, _feedback} when is_boolean(valid) and is_list(counts) ->
            {valid, ensure_three_counts(counts)}
          _ -> {false, [0, 0, 0]}
        end
    end

    {:noreply,
     socket
     |> assign(:task, task)
     |> assign(:live_action, :edit)
     |> assign(:debug_info, %{
       last_validation: DateTime.utc_now(),
       validation_count: 0,
       is_valid: is_valid,
       syllable_counts: syllable_counts
     })
     |> assign(:page_title, "Edit Haiku")}
  end

  @impl true
  def handle_event("new-haiku", _params, socket) do
    {:noreply,
     socket
     |> assign(:task, nil)
     |> assign(:live_action, :index)
     |> assign(:page_title, "TodoHaiku")}
  end

  # Helper function to ensure syllable_counts is always a 3-element array
  defp ensure_three_counts(counts) when is_list(counts) do
    case length(counts) do
      0 -> [0, 0, 0]
      1 -> counts ++ [0, 0]
      2 -> counts ++ [0]
      3 -> counts
      _ -> Enum.take(counts, 3)
    end
  end

  defp ensure_three_counts(_), do: [0, 0, 0]

  defp detect_over_limit(previous_counts, current_counts) do
    # Haiku syllable limits per line: [5, 7, 5]
    limits = [5, 7, 5]

    # Check if any line went from at-or-under limit to over limit
    Enum.with_index(previous_counts)
    |> Enum.find_value(fn {prev_count, index} ->
      current_count = Enum.at(current_counts, index)
      limit = Enum.at(limits, index)

      if prev_count <= limit && current_count > limit do
        index # Return the line index that went over limit
      else
        nil
      end
    end)
  end
end
