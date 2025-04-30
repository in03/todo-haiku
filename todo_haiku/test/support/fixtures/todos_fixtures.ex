defmodule TodoHaiku.TodosFixtures do
  @moduledoc """
  This module defines test helpers for creating
  entities via the `TodoHaiku.Todos` context.
  """

  @doc """
  Generate a task.
  """
  def task_fixture(attrs \\ %{}) do
    {:ok, task} =
      attrs
      |> Enum.into(%{
        content: "some content",
        is_completed: true,
        status: "some status",
        title: "some title"
      })
      |> TodoHaiku.Todos.create_task()

    task
  end
end
