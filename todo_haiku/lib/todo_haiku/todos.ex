defmodule TodoHaiku.Todos do
  @moduledoc """
  The Todos context.
  """

  import Ecto.Query, warn: false
  alias TodoHaiku.Repo

  alias TodoHaiku.Todos.Task

  @doc """
  Returns the list of tasks.

  ## Examples

      iex> list_tasks()
      [%Task{}, ...]

  """
  def list_tasks do
    Task
    |> order_by([t], [t.status, t.position, t.inserted_at])
    |> Repo.all()
  end

  @doc """
  Returns the list of tasks for a specific status, ordered by position.
  """
  def list_tasks_by_status(status) do
    Task
    |> where([t], t.status == ^status)
    |> order_by([t], [t.position, t.inserted_at])
    |> Repo.all()
  end

  @doc """
  Gets a single task.

  Raises `Ecto.NoResultsError` if the Task does not exist.

  ## Examples

      iex> get_task!(123)
      %Task{}

      iex> get_task!(456)
      ** (Ecto.NoResultsError)

  """
  def get_task!(id), do: Repo.get!(Task, id)

  @doc """
  Creates a task.

  ## Examples

      iex> create_task(%{field: value})
      {:ok, %Task{}}

      iex> create_task(%{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def create_task(attrs \\ %{}) do
    # Find the maximum position for this status
    status = Map.get(attrs, "status", "open")
    max_position = get_max_position(status)

    # Set the task position to be at the end
    attrs = Map.put(attrs, "position", max_position + 1)

    %Task{}
    |> Task.changeset(attrs)
    |> Repo.insert()
  end

  @doc """
  Updates a task.

  ## Examples

      iex> update_task(task, %{field: new_value})
      {:ok, %Task{}}

      iex> update_task(task, %{field: bad_value})
      {:error, %Ecto.Changeset{}}

  """
  def update_task(%Task{} = task, attrs) do
    # Check if the status is changing
    new_status = Map.get(attrs, "status")
    old_status = task.status

    # If status is changing, handle positioning
    attrs = if new_status && new_status != old_status do
      # Get the maximum position for the new status
      max_position = get_max_position(new_status)
      # Put the task at the end of its new status column
      Map.put(attrs, "position", max_position + 1)
    else
      attrs
    end

    task
    |> Task.changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Moves a task to a specific position within a status.
  """
  def reposition_task(%Task{} = task, new_status, to_position) do
    old_status = task.status
    old_position = task.position

    # Start a transaction to update all positions
    Repo.transaction(fn ->
      # If status is changing, update positions in both columns
      if new_status != old_status do
        # Decrement positions for tasks after the current position in the old status
        from(t in Task,
          where: t.status == ^old_status and t.position > ^old_position,
          update: [inc: [position: -1]]
        ) |> Repo.update_all([])

        # Increment positions for tasks at or after the new position in the new status
        from(t in Task,
          where: t.status == ^new_status and t.position >= ^to_position,
          update: [inc: [position: 1]]
        ) |> Repo.update_all([])
      else
        # Moving within the same status column
        if old_position < to_position do
          # Moving down - decrement tasks in between
          from(t in Task,
            where: t.status == ^old_status and t.position > ^old_position and t.position <= ^to_position,
            update: [inc: [position: -1]]
          ) |> Repo.update_all([])
        else
          # Moving up - increment tasks in between
          from(t in Task,
            where: t.status == ^old_status and t.position >= ^to_position and t.position < ^old_position,
            update: [inc: [position: 1]]
          ) |> Repo.update_all([])
        end
      end

      # Update the task with its new status and position
      task
      |> Task.changeset(%{"status" => new_status, "position" => to_position})
      |> Repo.update!()
    end)
  end

  @doc """
  Deletes a task.

  ## Examples

      iex> delete_task(task)
      {:ok, %Task{}}

      iex> delete_task(task)
      {:error, %Ecto.Changeset{}}

  """
  def delete_task(%Task{} = task) do
    # Decrement positions for tasks after the deleted task
    Repo.transaction(fn ->
      # Delete the task
      result = Repo.delete(task)

      # Update positions of remaining tasks in the same status
      from(t in Task,
        where: t.status == ^task.status and t.position > ^task.position,
        update: [inc: [position: -1]]
      ) |> Repo.update_all([])

      result
    end)
  end

  @doc """
  Returns an `%Ecto.Changeset{}` for tracking task changes.

  ## Examples

      iex> change_task(task)
      %Ecto.Changeset{data: %Task{}}

  """
  def change_task(%Task{} = task, attrs \\ %{}) do
    Task.changeset(task, attrs)
  end

  # Private helper functions

  # Get the maximum position for a given status
  defp get_max_position(status) do
    query = from t in Task,
              where: t.status == ^status,
              select: max(t.position)

    case Repo.one(query) do
      nil -> -1  # No tasks in this status yet
      max_pos -> max_pos
    end
  end
end
