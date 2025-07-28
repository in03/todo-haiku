defmodule TodoHaiku.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      TodoHaikuWeb.Telemetry,
      TodoHaiku.Repo,
      {DNSCluster, query: Application.get_env(:todo_haiku, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: TodoHaiku.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: TodoHaiku.Finch},
      # Start a worker by calling: TodoHaiku.Worker.start_link(arg)
      # {TodoHaiku.Worker, arg},
      # Start to serve requests, typically the last entry
      TodoHaikuWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: TodoHaiku.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    TodoHaikuWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
