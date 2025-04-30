defmodule TodoHaiku.Repo do
  use Ecto.Repo,
    otp_app: :todo_haiku,
    adapter: Ecto.Adapters.SQLite3
end
