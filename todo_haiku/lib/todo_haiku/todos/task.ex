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
    field :is_completed, :boolean, default: false
    field :syllable_counts, {:array, :integer}, virtual: true
    field :is_valid_haiku, :boolean, virtual: true
    field :feedback, :string, virtual: true

    timestamps(type: :utc_datetime)
  end

  # Implementing Access behaviour callbacks
  @impl true
  def fetch(struct, key) do
    Map.fetch(struct, key)
  end

  @impl true
  def get(struct, key, default \\ nil) do
    Map.get(struct, key, default)
  end

  @impl true
  def get_and_update(struct, key, fun) do
    Map.get_and_update(struct, key, fun)
  end

  @impl true
  def pop(struct, key) do
    {get(struct, key), Map.drop(struct, [key])}
  end

  @doc false
  def changeset(task, attrs) do
    IO.puts("Task changeset called with attrs: #{inspect(attrs)}")

    task
    |> cast(attrs, [:title, :content, :is_completed, :status])
    |> validate_required([:title, :content])
    |> validate_status()
    |> validate_haiku()
    |> validate_haiku_is_valid()
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

  defp validate_haiku(changeset) do
    content = get_field(changeset, :content)
    IO.puts("Validating haiku content: #{inspect(content)}")

    if is_nil(content) do
      changeset
    else
      {is_valid, syllable_counts, feedback} = TodoHaiku.HaikuValidator.validate_haiku(content)
      IO.puts("Schema validation result: is_valid=#{inspect(is_valid)}, syllable_counts=#{inspect(syllable_counts)}")

      changeset
      |> put_change(:syllable_counts, syllable_counts)
      |> put_change(:is_valid_haiku, is_valid)
      |> put_change(:feedback, feedback)
    end
  end

  defp validate_haiku_is_valid(changeset) do
    # Get the validation status
    is_valid_haiku = get_field(changeset, :is_valid_haiku)
    IO.puts("validate_haiku_is_valid called, is_valid_haiku=#{inspect(is_valid_haiku)}")

    if is_valid_haiku == false do
      IO.puts("Adding error: haiku is not valid")
      add_error(changeset, :content, "must be a valid haiku with 5-7-5 syllable pattern")
    else
      changeset
    end
  end
end
