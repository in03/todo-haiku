defmodule TodoHaiku.Repo.Migrations.AddOauthFieldsToUsers do
  use Ecto.Migration

  def change do
    alter table(:users) do
      add :provider, :string
      add :provider_uid, :string
      add :avatar_url, :string
      add :github_username, :string
      add :name, :string
      add :access_token, :string, size: 512
    end

    create index(:users, [:provider, :provider_uid], unique: true)
  end
end
