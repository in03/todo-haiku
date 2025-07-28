defmodule TodoHaiku.Repo.Migrations.CreateTasks do
  use Ecto.Migration

  def change do
    create table(:tasks, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :title, :string
      add :content, :text
      add :is_completed, :boolean, default: false, null: false
      add :status, :string

      timestamps(type: :utc_datetime)
    end
  end
end
