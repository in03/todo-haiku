defmodule TodoHaikuWeb.AuthController do
  use TodoHaikuWeb, :controller
  plug Ueberauth

  alias TodoHaiku.Accounts
  alias TodoHaikuWeb.UserAuth

  def request(conn, _params) do
    # This route is handled by Ueberauth
    conn
    |> put_flash(:info, "Redirecting to GitHub...")
    |> redirect(to: ~p"/")
  end

  def callback(%{assigns: %{ueberauth_failure: _fails}} = conn, _params) do
    conn
    |> put_flash(:error, "Failed to authenticate with GitHub.")
    |> redirect(to: ~p"/")
  end

  def callback(%{assigns: %{ueberauth_auth: auth}} = conn, _params) do
    case Accounts.get_or_create_user_from_github(%{
      provider: auth.provider,
      uid: auth.uid,
      email: auth.info.email,
      name: auth.info.name,
      nickname: auth.info.nickname,
      avatar: auth.info.image,
      token: auth.credentials.token
    }) do
      {:ok, user} ->
        conn
        |> UserAuth.log_in_user(user)
        |> put_flash(:info, "Successfully authenticated with GitHub!")
        |> redirect(to: ~p"/tasks")

      {:error, reason} ->
        conn
        |> put_flash(:error, "Authentication with GitHub failed: #{inspect(reason)}")
        |> redirect(to: ~p"/")
    end
  end

  def delete(conn, _params) do
    conn
    |> UserAuth.log_out_user()
    |> put_flash(:info, "Logged out successfully.")
    |> redirect(to: ~p"/")
  end
end
