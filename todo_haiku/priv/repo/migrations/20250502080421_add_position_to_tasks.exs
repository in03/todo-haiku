defmodule TodoHaiku.Repo.Migrations.AddPositionToTasks do
  use Ecto.Migration

  def change do
    alter table(:tasks) do
      add :position, :integer, default: 0
    end

    # Create an index for faster sorting
    create index(:tasks, [:status, :position])

    # Run a function to initialize position values
    execute """
    UPDATE tasks
    SET position = (
      SELECT COUNT(*)
      FROM tasks t2
      WHERE t2.status = tasks.status AND t2.inserted_at <= tasks.inserted_at
    ) - 1
    """
  end
end
