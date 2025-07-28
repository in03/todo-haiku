defmodule TodoHaikuWeb.PageController do
  use TodoHaikuWeb, :controller

  def home(conn, _params) do
    # If user is already logged in, redirect to tasks
    if conn.assigns.current_user do
      redirect(conn, to: ~p"/tasks")
    else
      # Show the welcome page for non-authenticated users
      render(conn, :home)
    end
  end
end
