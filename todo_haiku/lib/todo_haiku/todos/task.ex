defmodule TodoHaiku.Todos.Task do
  use Ecto.Schema
  import Ecto.Changeset
  @behaviour Access

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "tasks" do
    field :status, :string
    field :title, :string
    field :content, :string
    field :position, :integer, default: 0
    field :syllable_counts, {:array, :integer}, virtual: true
    field :is_valid_haiku, :boolean, virtual: true
    field :feedback, :string, virtual: true

    timestamps(type: :utc_datetime)
  end

  # Implementing Access behaviour callbacks
  @impl Access
  def fetch(struct, key) do
    Map.fetch(struct, key)
  end

  def get(struct, key, default \\ nil) do
    Map.get(struct, key, default)
  end

  @impl Access
  def get_and_update(struct, key, fun) do
    Map.get_and_update(struct, key, fun)
  end

  @impl Access
  def pop(struct, key) do
    {get(struct, key), Map.drop(struct, [key])}
  end

  @doc false
  def changeset(task, attrs) do
    IO.puts("Task changeset called with attrs: #{inspect(attrs)}")

    task
    |> cast(attrs, [:title, :content, :status, :position, :syllable_counts, :is_valid_haiku, :feedback])
    |> validate_required([:title, :content])
    |> validate_status()
  end

  defp validate_status(changeset) do
    case get_field(changeset, :status) do
      nil ->
        put_change(changeset, :status, "open")
      status when status in ["open", "doing", "done", "blocked"] ->
        changeset
      _ ->
        add_error(changeset, :status, "must be one of: open, doing, done, blocked")
    end
  end


end
