# Todo Haiku - Deployment Guide

This guide explains how to deploy Todo Haiku to Fly.io.

## Prerequisites

1. [Install Fly CLI](https://fly.io/docs/hands-on/install-flyctl/)
2. Sign up for a Fly.io account
3. Login to Fly.io CLI: `fly auth login`
4. [GitHub OAuth App](https://github.com/settings/developers) for production

## Local Setup with Tailscale Funnel

Before deploying to Fly.io, you can test the application locally with GitHub OAuth using Tailscale Funnel:

1. Install [Tailscale](https://tailscale.com/download) on your development machine

2. Enable Tailscale Funnel to expose your local Phoenix server:
   ```bash
   tailscale funnel 4000 on
   ```

3. Get your Tailscale machine name:
   ```bash
   tailscale status
   ```
   Note your machine's hostname (e.g., `your-machine.your-tailnet.ts.net`)

4. Set up GitHub OAuth for local development:
   - Create a GitHub OAuth application at https://github.com/settings/developers
   - Set Authorization callback URL to `https://your-machine.your-tailnet.ts.net/auth/github/callback`
   - Note your Client ID and Client Secret

5. Set environment variables:
   ```bash
   # For bash/zsh
   export GITHUB_CLIENT_ID=your_github_client_id
   export GITHUB_CLIENT_SECRET=your_github_client_secret

   # For Windows PowerShell
   $env:GITHUB_CLIENT_ID = "your_github_client_id"
   $env:GITHUB_CLIENT_SECRET = "your_github_client_secret"
   ```

6. Start your Phoenix server:
   ```bash
   mix phx.server
   ```

7. Access your application at `https://your-machine.your-tailnet.ts.net`

## Deploy to Fly.io

1. Generate a secret key base:
   ```bash
   mix phx.gen.secret
   ```

2. Set up GitHub OAuth for production:
   - Create a GitHub OAuth application at https://github.com/settings/developers
   - Set Authorization callback URL to `https://todo-haiku.fly.dev/auth/github/callback`
   - Note your Client ID and Client Secret

3. Launch the app on Fly (first time only):
   ```bash
   fly launch --name todo-haiku --region <your-preferred-region>
   ```
   
   - Answer "No" when asked if you want to create a Postgres database
   - Answer "Yes" when asked about deploying now

4. Create a volume for persistent SQLite data:
   ```bash
   fly volumes create todo_haiku_data --size 1 --region <your-region>
   ```

5. Set required secrets:
   ```bash
   fly secrets set SECRET_KEY_BASE=<your-generated-secret>
   fly secrets set GITHUB_CLIENT_ID=<your-github-client-id>
   fly secrets set GITHUB_CLIENT_SECRET=<your-github-client-secret>
   ```

6. Deploy the application:
   ```bash
   fly deploy
   ```

7. Open the application:
   ```bash
   fly open
   ```

## Continuous Deployment

This project includes a GitHub Actions workflow for continuous deployment. To set it up:

1. Add the following secrets to your GitHub repository:
   - `FLY_API_TOKEN`: Generate with `fly auth token`
   - `SECRET_KEY_BASE`: Your generated secret key base
   - `GITHUB_CLIENT_ID`: Your GitHub OAuth Client ID
   - `GITHUB_CLIENT_SECRET`: Your GitHub OAuth Client Secret

2. Push to the main branch to trigger automatic deployment.

## Alternative Deployment Option: Cloudflare Tunnel

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) provides a secure way to connect your local application to the internet:

1. Install cloudflared: `brew install cloudflare/cloudflare/cloudflared`
2. Authenticate: `cloudflared tunnel login`
3. Create a tunnel: `cloudflared tunnel create todo-haiku`
4. Configure the tunnel in `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <your-tunnel-id>
   credentials-file: /path/to/credentials.json
   ingress:
     - hostname: todo-haiku.yourdomain.com
       service: http://localhost:4000
     - service: http_status:404
   ```
5. Route DNS to your tunnel: `cloudflared tunnel route dns <your-tunnel-id> todo-haiku.yourdomain.com`
6. Start the tunnel: `cloudflared tunnel run`
7. Update your GitHub OAuth callback URL to match your Cloudflare domain

## Troubleshooting

### GitHub OAuth Issues

If you encounter OAuth errors like `no case clause matching: nil`:

1. Verify your GitHub OAuth credentials are correctly set:
   ```bash
   # Check if environment variables are set
   echo $GITHUB_CLIENT_ID
   echo $GITHUB_CLIENT_SECRET
   ```

2. Make sure the callback URL in your GitHub OAuth app matches your actual URL:
   - For Tailscale: `https://your-machine.your-tailnet.ts.net/auth/github/callback`
   - For Fly.io: `https://todo-haiku.fly.dev/auth/github/callback`
   - For Cloudflare: `https://todo-haiku.yourdomain.com/auth/github/callback`

3. Restart your Phoenix server after setting environment variables 