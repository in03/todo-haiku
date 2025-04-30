defmodule TodoHaikuWeb.PageController do
  use TodoHaikuWeb, :controller

  def home(conn, _params) do
    # Redirect to the tasks page
    redirect(conn, to: ~p"/tasks")
  end
end
