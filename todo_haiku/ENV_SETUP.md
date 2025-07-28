# Environment Variables for Todo Haiku

This guide explains how to manage environment variables and secrets for the Todo Haiku application.

## Recommended Approach: Config Secret Files

For development, we use secret config files that aren't checked into source control:

### Setup Instructions

1. A `config/dev.secret.exs` file has been created with placeholder values:

```elixir
import Config

# GitHub OAuth configuration
# Replace these placeholder values with your actual GitHub OAuth credentials
config :ueberauth, Ueberauth.Strategy.Github.OAuth,
  client_id: "github_client_id_placeholder",
  client_secret: "github_client_secret_placeholder"
```

2. Edit this file to replace the placeholder values with your actual credentials.

3. The `.gitignore` file has been updated to exclude these secret files:

```
/config/*.secret.exs
```

4. The `config/dev.exs` file already imports this secret configuration file:

```elixir
# Import the secret configuration file if it exists
if File.exists?("config/dev.secret.exs") do
  import_config "dev.secret.exs"
end
```

This approach keeps sensitive information out of your Git repository while still making it easy to configure your development environment.

## Deployment to Fly.io

For production deployment on Fly.io, the application continues to use environment variables set through Fly.io secrets. This approach remains unchanged and works alongside our development secrets approach.

To set secrets for production:

```bash
fly secrets set GITHUB_CLIENT_ID=your_github_client_id
fly secrets set GITHUB_CLIENT_SECRET=your_github_client_secret
fly secrets set SECRET_KEY_BASE=your_secret_key_base
```

The `config/runtime.exs` file is already configured to read these environment variables in production.

## Alternative Approaches (if needed)

### Windows PowerShell

To set environment variables for your current PowerShell session:

```powershell
$env:GITHUB_CLIENT_ID = "your_github_client_id"
$env:GITHUB_CLIENT_SECRET = "your_github_client_secret"
```

### Windows with VSCode

You can create a `.vscode/launch.json` file to set environment variables when debugging:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "mix_task",
      "name": "mix phx.server",
      "request": "launch",
      "task": "phx.server",
      "env": {
        "GITHUB_CLIENT_ID": "your_github_client_id",
        "GITHUB_CLIENT_SECRET": "your_github_client_secret"
      }
    }
  ]
}
```

## Required Environment Variables

The following environment variables/settings are required:

For development (in `config/dev.secret.exs`):
- `client_id` - Your GitHub OAuth application client ID
- `client_secret` - Your GitHub OAuth application client secret

For production (as Fly.io secrets):
- `GITHUB_CLIENT_ID` - Your GitHub OAuth application client ID
- `GITHUB_CLIENT_SECRET` - Your GitHub OAuth application client secret
- `SECRET_KEY_BASE` - Secret key for encrypting cookies (generate with `mix phx.gen.secret`) 