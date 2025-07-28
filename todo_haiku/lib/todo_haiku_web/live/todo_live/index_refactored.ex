defmodule TodoHaikuWeb.TodoLive.IndexRefactored do
  @moduledoc """
  Refactored TodoLive.Index using modern component architecture.
  This demonstrates the cleaner, more maintainable approach.
  """
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
     |> assign(:search_term, nil)
     |> assign(:template_task, %Task{})
     |> assign(:changeset, Todos.change_task(%Task{}))
     |> assign(:form, nil)
     |> assign(:trigger_submit, false)
     |> assign(:validation_state, initial_validation_state())
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
    status = Map.get(params, "status", "open")

    task = %Task{
      is_valid_haiku: false,
      syllable_counts: [0, 0, 0],
      feedback: "Enter your haiku",
      status: status
    }

    changeset = Todos.change_task(task)

    socket
    |> assign(:page_title, "New Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(changeset))
  end

  defp apply_action(socket, :edit, %{"id" => id}) do
    task = Todos.get_task!(id)
    {is_valid, syllable_counts, feedback} = validate_task_content(task.content)

    changeset =
      task
      |> Todos.change_task()
      |> put_validation_results(is_valid, syllable_counts, feedback)

    validation_state = %{
      last_validation: DateTime.utc_now(),
      validation_count: 1,
      is_valid: is_valid,
      syllable_counts: syllable_counts
    }

    socket
    |> assign(:page_title, "Edit Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(changeset))
    |> assign(:validation_state, validation_state)
  end

  @impl true
  def handle_event("search", %{"value" => search_term}, socket) do
    {:noreply, assign(socket, :search_term, search_term)}
  end

  @impl true
  def handle_event("search", params, socket) do
    search_term = Map.get(params, "search", "")
    {:noreply, assign(socket, :search_term, search_term)}
  end

  @impl true
  def handle_event("delete", %{"id" => id}, socket) do
    task = Todos.get_task!(id)
    {:ok, _} = Todos.delete_task(task)

    tasks = Todos.list_tasks()
    {:noreply, assign(socket, :tasks, tasks)}
  end

  @impl true
  def handle_event("validate", %{"task" => task_params}, socket) do
    content = Map.get(task_params, "content", "")

    # Cancel any existing validation task
    cancel_existing_validation_task(socket)

    if content == "" do
      handle_empty_content_validation(socket, task_params)
    else
      handle_async_validation(socket, task_params, content)
    end
  end

  @impl true
  def handle_event("save", %{"task" => task_params}, socket) do
    if socket.assigns.validation_state.is_valid do
      save_task(socket, socket.assigns.live_action, task_params)
    else
      handle_invalid_save_attempt(socket, task_params)
    end
  end

  @impl true
  def handle_event("generate_template", _, socket) do
    {title, content} = get_random_haiku_template()
    task_params = %{"title" => title, "content" => content}

    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    {is_valid, syllable_counts, feedback} = HaikuValidator.validate_haiku(content)
    changeset = put_validation_results(changeset, is_valid, syllable_counts, feedback)

    validation_state = update_validation_state(socket.assigns.validation_state, is_valid, syllable_counts)

    {:noreply,
      socket
      |> assign(:form, to_form(changeset))
      |> assign(:validation_state, validation_state)
    }
  end

  @impl true
  def handle_event("task-moved", %{"id" => id, "status" => new_status, "position" => position}, socket) do
    task = Todos.get_task!(id)

    case Todos.reposition_task(task, new_status, position) do
      {:ok, _} ->
        tasks = Todos.list_tasks()
        {:noreply, assign(socket, :tasks, tasks)}

      {:error, _reason} ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_info({ref, result}, socket) do
    if socket.assigns[:validation_task] && ref == socket.assigns[:validation_task].ref do
      Process.demonitor(ref, [:flush])
      handle_async_validation_result(socket, result)
    else
      {:noreply, socket}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, _reason}, socket) do
    if socket.assigns[:validation_task] && ref == socket.assigns[:validation_task].ref do
      {:noreply, cleanup_validation_task(socket)}
    else
      {:noreply, socket}
    end
  end

  @impl true
  def handle_info(_message, socket), do: {:noreply, socket}

  # Private helper functions

  defp initial_validation_state do
    %{
      last_validation: nil,
      validation_count: 0,
      is_valid: false,
      syllable_counts: [0, 0, 0]
    }
  end

  defp validate_task_content(""), do: {false, [0, 0, 0], "A haiku is required."}
  defp validate_task_content(content), do: TodoHaiku.HaikuValidator.validate_haiku(content)

  defp put_validation_results(changeset, is_valid, syllable_counts, feedback) do
    changeset
    |> Ecto.Changeset.put_change(:is_valid_haiku, is_valid)
    |> Ecto.Changeset.put_change(:syllable_counts, syllable_counts)
    |> Ecto.Changeset.put_change(:feedback, feedback)
  end

  defp update_validation_state(current_state, is_valid, syllable_counts) do
    %{
      last_validation: DateTime.utc_now(),
      validation_count: current_state.validation_count + 1,
      is_valid: is_valid,
      syllable_counts: syllable_counts
    }
  end

  defp cancel_existing_validation_task(socket) do
    if task_ref = socket.assigns[:validation_task] do
      Process.exit(task_ref.pid, :brutal_kill)
    end
  end

  defp handle_empty_content_validation(socket, task_params) do
    validation_state = update_validation_state(socket.assigns.validation_state, false, [0, 0, 0])

    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    {:noreply,
      socket
      |> assign(:validation_state, validation_state)
      |> assign(:form, to_form(changeset))
      |> assign(:validation_task, nil)
      |> assign(:pending_content, nil)
    }
  end

  defp handle_async_validation(socket, task_params, content) do
    task = TodoHaiku.BigPhoneyClient.count_syllables_haiku_async(content)

    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    validation_state = update_validation_state(socket.assigns.validation_state, false, [0, 0, 0])

    {:noreply,
      socket
      |> assign(:validation_task, task)
      |> assign(:pending_content, content)
      |> assign(:pending_task_params, task_params)
      |> assign(:form, to_form(changeset))
      |> assign(:validation_state, validation_state)
    }
  end

  defp handle_async_validation_result(socket, result) do
    case result do
      {:ok, %{"lines" => lines}} ->
        syllable_counts = process_syllable_response(lines)
        is_valid = syllable_counts == [5, 7, 5]
        feedback = if is_valid, do: "Perfect haiku! You're a natural poet.", else: "Not quite a haiku yet. Keep adjusting your words."

        validation_state = update_validation_state(socket.assigns.validation_state, is_valid, syllable_counts)

        task_params = Map.merge(socket.assigns.pending_task_params, %{
          "is_valid_haiku" => is_valid,
          "syllable_counts" => syllable_counts,
          "feedback" => feedback
        })

        changeset =
          socket.assigns.task
          |> Todos.change_task(task_params)
          |> Map.put(:action, :validate)

        {:noreply,
          socket
          |> assign(:form, to_form(changeset))
          |> assign(:validation_state, validation_state)
          |> cleanup_validation_task()
        }

      {:error, _reason} ->
        {:noreply, cleanup_validation_task(socket)}
    end
  end

  defp process_syllable_response(lines) do
    syllable_counts = Enum.map(lines, & &1["syllables"])

    case length(syllable_counts) do
      0 -> [0, 0, 0]
      1 -> syllable_counts ++ [0, 0]
      2 -> syllable_counts ++ [0]
      3 -> syllable_counts
      _ -> Enum.take(syllable_counts, 3)
    end
  end

  defp cleanup_validation_task(socket) do
    socket
    |> assign(:validation_task, nil)
    |> assign(:pending_content, nil)
    |> assign(:pending_task_params, nil)
  end

  defp handle_invalid_save_attempt(socket, task_params) do
    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)
      |> Ecto.Changeset.add_error(:content, "must be a valid haiku with 5-7-5 syllable pattern")

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

  defp get_random_haiku_template do
    examples = [
      {"Do Laundry", "High piles of laundry\nGetting so tired of this\nWhen will it all end?"},
      {"Morning Exercise", "Early morning run\nFeet pounding on the pavement\nStrength builds with each step"},
      {"Study Session", "Books spread on the desk\nKnowledge flows through fingertips\nMind grows like a tree"},
      {"Self Care Evening", "Candles flicker soft\nRelaxation washes through\nTime just for myself"},
      {"Grocery Shopping", "Empty pantry calls\nWheels squeak along tile floors\nFridge now overflows"}
    ]

    Enum.random(examples)
  end
end
