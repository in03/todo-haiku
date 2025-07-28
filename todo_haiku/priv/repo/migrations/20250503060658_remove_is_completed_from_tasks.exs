defmodule TodoHaiku.Repo.Migrations.RemoveIsCompletedFromTasks do
  use Ecto.Migration

  def change do
    alter table(:tasks) do
      remove :is_completed
    end
  end
end
