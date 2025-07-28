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
     |> assign(:template_task, %Task{})
     |> assign(:changeset, Todos.change_task(%Task{}))
     |> assign(:form, nil)
     |> assign(:trigger_submit, false)
     |> assign(:debug_info, %{
         last_validation: nil,
         validation_count: 0,
         is_valid: false,
         syllable_counts: [0, 0, 0]
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
    |> assign(:form, to_form(Todos.change_task(%Task{})))
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

  @impl true
  def handle_event("search", %{"value" => search_term}, socket) do
    # Update the search term in the socket assigns
    {:noreply, assign(socket, :search_term, search_term)}
  end

  # When no value key is found, this might be coming directly from the input
  @impl true
  def handle_event("search", params, socket) do
    search_term = Map.get(params, "search", "")
    {:noreply, assign(socket, :search_term, search_term)}
  end

  @impl true
  def handle_event("filter", %{"filter" => filter}, socket) do
    tasks = case filter do
      "all" -> Todos.list_tasks()
      "active" -> Todos.list_tasks() |> Enum.filter(fn task -> !task.is_completed end)
      "completed" -> Todos.list_tasks() |> Enum.filter(fn task -> task.is_completed end)
      _ -> Todos.list_tasks()
    end

    {:noreply, socket |> assign(:tasks, tasks) |> assign(:filter, filter)}
  end

  @impl true
  def handle_event("delete", %{"id" => id}, socket) do
    task = Todos.get_task!(id)
    {:ok, _} = Todos.delete_task(task)

    tasks = case socket.assigns.filter do
      "all" -> Todos.list_tasks()
      "active" -> Todos.list_tasks() |> Enum.filter(fn task -> !task.is_completed end)
      "completed" -> Todos.list_tasks() |> Enum.filter(fn task -> task.is_completed end)
      _ -> Todos.list_tasks()
    end

    {:noreply, assign(socket, :tasks, tasks)}
  end

  # @impl true
  # def handle_event("toggle", %{"id" => id}, socket) do
  #   task = Todos.get_task!(id)
  #   {:ok, _updated_task} = Todos.update_task(task, %{is_completed: !task.is_completed})
  #
  #   tasks = case socket.assigns.filter do
  #     "all" -> Todos.list_tasks()
  #     "active" -> Todos.list_tasks() |> Enum.filter(fn t -> !t.is_completed end)
  #     "completed" -> Todos.list_tasks() |> Enum.filter(fn t -> t.is_completed end)
  #     _ -> Todos.list_tasks()
  #   end
  #
  #   {:noreply, assign(socket, :tasks, tasks)}
  # end

  @impl true
  def handle_event("validate", %{"task" => task_params}, socket) do
    # Log for debugging
    IO.puts("Validating task with params: #{inspect(task_params)}")

    # Get the current validation count and increment it
    current_count = socket.assigns.debug_info.validation_count || 0

    # Create a base task with the content from params
    content = Map.get(task_params, "content", "")

    # Log content
    IO.puts("Content to validate: #{inspect(content)}")

    # Cancel any existing validation task
    if task_ref = socket.assigns[:validation_task] do
      Task.shutdown(task_ref, :brutal_kill)
    end

    if is_nil(content) or content == "" do
      # Handle empty content immediately
      debug_info = %{
        last_validation: DateTime.utc_now(),
        validation_count: current_count + 1,
        is_valid: false,
        syllable_counts: [0, 0, 0]
      }

      changeset =
        socket.assigns.task
        |> Todos.change_task(task_params)
        |> Map.put(:action, :validate)

      {:noreply,
        socket
        |> assign(:debug_info, debug_info)
        |> assign(:form, to_form(changeset))
        |> assign(:validation_task, nil)
        |> assign(:pending_content, nil)
      }
    else
      # Start async validation using the new haiku endpoint
      IO.puts("Starting async validation for: #{inspect(content)}")
      task = TodoHaiku.BigPhoneyClient.count_syllables_haiku_async(content)
      IO.puts("Task started with ref: #{inspect(task.ref)}")

      # Store the task reference and content for later processing
      changeset =
        socket.assigns.task
        |> Todos.change_task(task_params)
        |> Map.put(:action, :validate)

      {:noreply,
        socket
        |> assign(:validation_task, task)
        |> assign(:pending_content, content)
        |> assign(:pending_task_params, task_params)
        |> assign(:form, to_form(changeset))
        |> assign(:debug_info, %{
          last_validation: DateTime.utc_now(),
          validation_count: current_count + 1,
          is_valid: false,
          syllable_counts: [0, 0, 0]
        })
      }
    end
  end

  @impl true
  def handle_event("save", %{"task" => task_params}, socket) do
    # Log save attempt
    IO.puts("Attempting to save task with params: #{inspect(task_params)}")

    # Use the cached validation state from debug_info
    is_valid = socket.assigns.debug_info.is_valid
    # Don't use syllable_counts in this function, prefix with underscore to ignore
    _syllable_counts = socket.assigns.debug_info.syllable_counts

    # Log validation status for save
    IO.puts("Is valid for save: #{inspect(is_valid)}")

    # Only proceed with save if haiku is valid
    if is_valid do
      IO.puts("Task is valid, proceeding with save")
      save_task(socket, socket.assigns.live_action, task_params)
    else
      IO.puts("Task is invalid, not saving")
      # Create a changeset with errors
      changeset =
        socket.assigns.task
        |> Todos.change_task(task_params)
        |> Map.put(:action, :validate)
        |> Ecto.Changeset.add_error(:content, "must be a valid haiku with 5-7-5 syllable pattern")

      {:noreply, assign(socket, :form, to_form(changeset))}
    end
  end

  @impl true
  def handle_event("generate_template", _, socket) do
    # Generate a template haiku and title
    examples = [
      {"Do Laundry",
       "High piles of laundry\nGetting so tired of this\nWhen will it all end?"},

      {"Morning Exercise",
       "Early morning run\nFeet pounding on the pavement\nStrength builds with each step"},

      {"Study Session",
       "Books spread on the desk\nKnowledge flows through fingertips\nMind grows like a tree"},

      {"Self Care Evening",
       "Candles flicker soft\nRelaxation washes through\nTime just for myself"},

      {"Grocery Shopping",
       "Empty pantry calls\nWheels squeak along tile floors\nFridge now overflows"}
    ]

    # Pick a random example
    {title, content} = Enum.random(examples)

    # Update the form with the template
    task_params = %{"title" => title, "content" => content}
    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    # Validate the haiku
    {is_valid, syllable_counts, feedback} = HaikuValidator.validate_haiku(content)

    changeset =
      changeset
      |> Ecto.Changeset.put_change(:is_valid_haiku, is_valid)
      |> Ecto.Changeset.put_change(:syllable_counts, syllable_counts)
      |> Ecto.Changeset.put_change(:feedback, feedback)

    # Update the debug info
    debug_info = %{
      last_validation: DateTime.utc_now(),
      validation_count: socket.assigns.debug_info.validation_count + 1,
      is_valid: is_valid,
      syllable_counts: syllable_counts
    }

    {:noreply,
      socket
      |> assign(:form, to_form(changeset))
      |> assign(:debug_info, debug_info)
    }
  end

  @impl true
  def handle_event("task-moved", %{"id" => id, "status" => new_status, "position" => position}, socket) do
    # Log the task move
    IO.puts("Task #{id} moved to #{new_status} at position #{position}")

    # Get the task
    task = Todos.get_task!(id)

    # Reposition the task
    case Todos.reposition_task(task, new_status, position) do
      {:ok, _} ->
        IO.puts("Task repositioned successfully: #{id}")

        # Refresh the task list
        tasks = case socket.assigns.filter do
          "all" -> Todos.list_tasks()
          "active" -> Todos.list_tasks() |> Enum.filter(fn t -> !t.is_completed end)
          "completed" -> Todos.list_tasks() |> Enum.filter(fn t -> t.is_completed end)
          _ -> Todos.list_tasks()
        end

        {:noreply, assign(socket, :tasks, tasks)}

      {:error, reason} ->
        IO.puts("Error repositioning task: #{inspect(reason)}")
        {:noreply, socket}
    end
  end

  defp save_task(socket, :edit, task_params) do
    case Todos.update_task(socket.assigns.task, task_params) do
      {:ok, _task} ->
        {:noreply,
         socket
         |> put_flash(:info, "Task updated successfully")
         |> push_navigate(to: ~p"/tasks")}

      {:error, %Ecto.Changeset{} = changeset} ->
        {:noreply, assign(socket, :form, to_form(changeset))}
    end
  end

  defp save_task(socket, :new, task_params) do
    case Todos.create_task(task_params) do
      {:ok, _task} ->
        {:noreply,
         socket
         |> put_flash(:info, "Task created successfully")
         |> push_navigate(to: ~p"/tasks")}

      {:error, %Ecto.Changeset{} = changeset} ->
        {:noreply, assign(socket, form: to_form(changeset))}
    end
  end

  @impl true
  def handle_info({ref, result}, socket) do
    IO.puts("Received async result with ref: #{inspect(ref)}")
    IO.puts("Current validation_task ref: #{inspect(socket.assigns[:validation_task])}")

    if socket.assigns[:validation_task] && ref == socket.assigns[:validation_task].ref do
      IO.puts("Processing async validation result")
      # Demonitor the completed task
      Process.demonitor(ref, [:flush])
    case result do
      {:ok, %{"lines" => lines}} ->
        # Extract syllable counts from the lines
        syllable_counts = Enum.map(lines, & &1["syllables"])

        # Pad to exactly 3 lines for haiku
        syllable_counts = case length(syllable_counts) do
          0 -> [0, 0, 0]
          1 -> syllable_counts ++ [0, 0]
          2 -> syllable_counts ++ [0]
          3 -> syllable_counts
          _ -> Enum.take(syllable_counts, 3)
        end

        # Check if it's a valid haiku (5-7-5 pattern)
        is_valid = syllable_counts == [5, 7, 5]

        # Generate feedback
        feedback = if is_valid do
          "Perfect haiku! You're a natural poet."
        else
          "Not quite a haiku yet. Keep adjusting your words."
        end

        # Update debug info with validation results
        debug_info = %{
          last_validation: DateTime.utc_now(),
          validation_count: socket.assigns.debug_info.validation_count,
          is_valid: is_valid,
          syllable_counts: syllable_counts
        }

        # Create changeset with validation results
        task_params = socket.assigns.pending_task_params
        |> Map.merge(%{
          "is_valid_haiku" => is_valid,
          "syllable_counts" => syllable_counts,
          "feedback" => feedback
        })

        changeset =
          socket.assigns.task
          |> Todos.change_task(task_params)
          |> Map.put(:action, :validate)

        IO.puts("Async validation complete: is_valid=#{is_valid}, syllable_counts=#{inspect(syllable_counts)}")

        {:noreply,
          socket
          |> assign(:form, to_form(changeset))
          |> assign(:debug_info, debug_info)
          |> assign(:validation_task, nil)
          |> assign(:pending_content, nil)
          |> assign(:pending_task_params, nil)
        }

      {:error, reason} ->
        IO.puts("Async validation failed: #{inspect(reason)}")

        # Handle error gracefully - show previous state
        {:noreply,
          socket
          |> assign(:validation_task, nil)
          |> assign(:pending_content, nil)
          |> assign(:pending_task_params, nil)
        }
      end
    else
      {:noreply, socket}
    end
  end

  # Handle task DOWN messages (when task crashes)
  @impl true
  def handle_info({:DOWN, ref, :process, _pid, _reason}, socket) do
    if socket.assigns[:validation_task] && ref == socket.assigns[:validation_task].ref do
      IO.puts("Validation task crashed, cleaning up")
      {:noreply,
        socket
        |> assign(:validation_task, nil)
        |> assign(:pending_content, nil)
        |> assign(:pending_task_params, nil)
      }
    else
      {:noreply, socket}
    end
  end

  # Handle any other messages
  @impl true
  def handle_info(_message, socket) do
    {:noreply, socket}
  end
end
