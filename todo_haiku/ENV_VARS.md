# Environment Variables for Todo Haiku

This document describes the environment variables required for the Todo Haiku application.

## Development

For local development, you need to set these environment variables:

```bash
# GitHub OAuth settings
# Create a GitHub OAuth application at https://github.com/settings/developers
# For local development with Tailscale, set callback URL to:
# https://your-machine-name.tailnet-name.ts.net/auth/github/callback
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

## Production (Fly.io)

For Fly.io deployment, you'll need:

```bash
# Generate with `mix phx.gen.secret`
SECRET_KEY_BASE=your_secret_key_base

# GitHub OAuth settings
# Create a GitHub OAuth application at https://github.com/settings/developers
# Set callback URL to: https://todo-haiku.fly.dev/auth/github/callback
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

## Setting Environment Variables

### Local Development

```bash
# For bash/zsh
export GITHUB_CLIENT_ID=your_github_client_id
export GITHUB_CLIENT_SECRET=your_github_client_secret

# For Windows PowerShell
$env:GITHUB_CLIENT_ID = "your_github_client_id"
$env:GITHUB_CLIENT_SECRET = "your_github_client_secret"
```

### Fly.io

```bash
# Set secrets on Fly.io
fly secrets set GITHUB_CLIENT_ID=your_github_client_id
fly secrets set GITHUB_CLIENT_SECRET=your_github_client_secret
fly secrets set SECRET_KEY_BASE=your_secret_key_base
``` 