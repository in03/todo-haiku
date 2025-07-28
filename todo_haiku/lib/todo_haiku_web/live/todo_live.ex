defmodule TodoHaikuWeb.TodoLive do
  use TodoHaikuWeb, :live_view

  alias TodoHaiku.Todos
  alias TodoHaiku.Todos.Task
  alias TodoHaiku.HaikuValidator

  @impl true
  def mount(_params, _session, socket) do
    # Verify the user is authenticated
    if socket.assigns[:current_user] do
      tasks = Todos.list_tasks()

      {:ok,
       socket
       |> assign(:tasks, tasks)
       |> assign(:filter, "all")
       |> assign(:template_task, %Task{})
       |> assign(:changeset, Todos.change_task(%Task{}))
       |> assign(:form, nil)
       |> assign(:page_title, "TodoHaiku")}
    else
      {:ok,
       socket
       |> put_flash(:error, "You must be logged in to access tasks.")
       |> redirect(to: ~p"/")}
    end
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

  defp apply_action(socket, :new, _params) do
    socket
    |> assign(:page_title, "New Haiku Task")
    |> assign(:task, %Task{})
    |> assign(:form, to_form(Todos.change_task(%Task{})))
  end

  defp apply_action(socket, :edit, %{"id" => id}) do
    task = Todos.get_task!(id)
    socket
    |> assign(:page_title, "Edit Haiku Task")
    |> assign(:task, task)
    |> assign(:form, to_form(Todos.change_task(task)))
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
    changeset =
      socket.assigns.task
      |> Todos.change_task(task_params)
      |> Map.put(:action, :validate)

    # Get virtual fields for feedback
    content = get_field_from_changeset(changeset, :content)
    {is_valid, syllable_counts, feedback} =
      if is_nil(content) do
        {false, [0, 0, 0], ""}
      else
        TodoHaiku.HaikuValidator.validate_haiku(content)
      end

    changeset =
      changeset
      |> Ecto.Changeset.put_change(:is_valid_haiku, is_valid)
      |> Ecto.Changeset.put_change(:syllable_counts, syllable_counts)
      |> Ecto.Changeset.put_change(:feedback, feedback)

    {:noreply, assign(socket, :form, to_form(changeset))}
  end

  @impl true
  def handle_event("save", %{"task" => task_params}, socket) do
    save_task(socket, socket.assigns.live_action, task_params)
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
