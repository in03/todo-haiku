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
     |> assign(:page_title, "Todo Haikus")}
  end

  @impl true
  def handle_params(params, _url, socket) do
    {:noreply, apply_action(socket, socket.assigns.live_action, params)}
  end

  defp apply_action(socket, :index, _params) do
    socket
    |> assign(:page_title, "Todo Haikus")
    |> assign(:task, nil)
    |> assign(:form, to_form(Todos.change_task(%Task{})))
  end

  defp apply_action(socket, :new, _params) do
    # Initialize with empty template for better UX
    task = %Task{
      is_valid_haiku: false,
      syllable_counts: [0, 0, 0],
      feedback: "Enter your haiku"
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

    socket
    |> assign(:page_title, "Edit Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(changeset))
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

  @impl true
  def handle_event("toggle", %{"id" => id}, socket) do
    task = Todos.get_task!(id)
    {:ok, _updated_task} = Todos.update_task(task, %{is_completed: !task.is_completed})

    tasks = case socket.assigns.filter do
      "all" -> Todos.list_tasks()
      "active" -> Todos.list_tasks() |> Enum.filter(fn t -> !t.is_completed end)
      "completed" -> Todos.list_tasks() |> Enum.filter(fn t -> t.is_completed end)
      _ -> Todos.list_tasks()
    end

    {:noreply, assign(socket, :tasks, tasks)}
  end

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

    # Validate haiku
    {is_valid, syllable_counts, feedback} =
      if is_nil(content) or content == "" do
        {false, [0, 0, 0], "A haiku is required."}
      else
        result = TodoHaiku.HaikuValidator.validate_haiku(content)
        # Log validation result
        IO.puts("Validation result: #{inspect(result)}")
        result
      end

    # Add validation results directly to task_params before creating the changeset
    task_params = Map.merge(task_params, %{
      "is_valid_haiku" => is_valid,
      "syllable_counts" => syllable_counts,
      "feedback" => feedback
    })

    # Now create the changeset with the updated params
    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    # Log the final changeset validity
    IO.puts("Is valid haiku: #{inspect(is_valid)}")
    IO.puts("Syllable counts: #{inspect(syllable_counts)}")
    IO.puts("Changeset errors: #{inspect(changeset.errors)}")

    # Update debug info
    debug_info = %{
      last_validation: DateTime.utc_now(),
      validation_count: current_count + 1,
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
  def handle_event("save", %{"task" => task_params}, socket) do
    # Log save attempt
    IO.puts("Attempting to save task with params: #{inspect(task_params)}")

    # Use the cached validation state from debug_info
    is_valid = socket.assigns.debug_info.is_valid
    syllable_counts = socket.assigns.debug_info.syllable_counts

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
    template = HaikuValidator.generate_template()

    # Update the form with the template
    task_params = %{"content" => template}
    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    # Validate the haiku
    {is_valid, syllable_counts, feedback} = HaikuValidator.validate_haiku(template)

    changeset =
      changeset
      |> Ecto.Changeset.put_change(:is_valid_haiku, is_valid)
      |> Ecto.Changeset.put_change(:syllable_counts, syllable_counts)
      |> Ecto.Changeset.put_change(:feedback, feedback)

    {:noreply, assign(socket, :form, to_form(changeset))}
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

  defp get_field_from_changeset(changeset, field) do
    Ecto.Changeset.get_field(changeset, field)
  end
end
