#!/usr/bin/env node

/**
 * This script helps set up Supabase for the Todo Haiku app
 * It will:
 * 1. Check if Supabase CLI is installed
 * 2. Initialize Supabase if needed
 * 3. Start Supabase locally
 * 4. Apply migrations
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ANSI color codes for prettier output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

// Helper function to execute commands and handle errors
function runCommand(command, errorMessage) {
  try {
    console.log(`${colors.dim}> ${command}${colors.reset}`);
    const output = execSync(command, { stdio: 'inherit' });
    return output;
  } catch (error) {
    console.error(`${colors.red}${errorMessage}${colors.reset}`);
    console.error(`${colors.dim}${error.message}${colors.reset}`);
    return null;
  }
}

// Check if Supabase CLI is installed
function checkSupabaseCLI() {
  try {
    execSync('supabase --version', { stdio: 'ignore' });
    console.log(`${colors.green}✓ Supabase CLI is installed${colors.reset}`);
    return true;
  } catch (error) {
    console.log(`${colors.yellow}⚠ Supabase CLI is not installed${colors.reset}`);
    console.log(`${colors.cyan}Installing Supabase CLI...${colors.reset}`);

    try {
      runCommand('bun install -g supabase', 'Failed to install Supabase CLI globally');
      console.log(`${colors.green}✓ Supabase CLI installed successfully${colors.reset}`);
      return true;
    } catch (installError) {
      console.error(`${colors.red}Failed to install Supabase CLI${colors.reset}`);
      console.log(`${colors.yellow}Please install it manually:${colors.reset}`);
      console.log(`${colors.cyan}bun install -g supabase${colors.reset}`);
      return false;
    }
  }
}

// Initialize Supabase if not already initialized
function initializeSupabase() {
  if (!fs.existsSync(path.join(process.cwd(), 'supabase', 'config.toml'))) {
    console.log(`${colors.cyan}Initializing Supabase...${colors.reset}`);
    runCommand('supabase init', 'Failed to initialize Supabase');
  } else {
    console.log(`${colors.green}✓ Supabase is already initialized${colors.reset}`);
  }
}

// Start Supabase locally
function startSupabase() {
  console.log(`${colors.cyan}Starting Supabase...${colors.reset}`);
  console.log(`${colors.yellow}This may take a few minutes on first run${colors.reset}`);
  runCommand('supabase start', 'Failed to start Supabase');
}

// Create .env file with Supabase credentials if it doesn't exist
function setupEnvFile() {
  const envPath = path.join(process.cwd(), '.env');
  const envExamplePath = path.join(process.cwd(), '.env.example');

  if (!fs.existsSync(envPath) && fs.existsSync(envExamplePath)) {
    console.log(`${colors.cyan}Creating .env file...${colors.reset}`);
    fs.copyFileSync(envExamplePath, envPath);

    // Get Supabase credentials
    try {
      const output = execSync('supabase status').toString();
      const urlMatch = output.match(/API URL: (http:\/\/[^\s]+)/);
      const keyMatch = output.match(/anon key: ([^\s]+)/);

      if (urlMatch && keyMatch) {
        let envContent = fs.readFileSync(envPath, 'utf8');
        envContent = envContent.replace('your-supabase-url', urlMatch[1]);
        envContent = envContent.replace('your-supabase-anon-key', keyMatch[1]);
        fs.writeFileSync(envPath, envContent);
        console.log(`${colors.green}✓ .env file created with Supabase credentials${colors.reset}`);
      }
    } catch (error) {
      console.log(`${colors.yellow}⚠ Could not automatically update .env with Supabase credentials${colors.reset}`);
      console.log(`${colors.yellow}Please update them manually from the Supabase dashboard${colors.reset}`);
    }
  } else if (fs.existsSync(envPath)) {
    console.log(`${colors.green}✓ .env file already exists${colors.reset}`);
  }
}

// Main function
async function main() {
  console.log(`${colors.bright}${colors.magenta}Todo Haiku - Supabase Setup${colors.reset}\n`);

  const cliInstalled = checkSupabaseCLI();
  if (!cliInstalled) {
    process.exit(1);
  }

  initializeSupabase();
  startSupabase();
  setupEnvFile();

  console.log(`\n${colors.green}${colors.bright}✓ Supabase setup complete!${colors.reset}`);
  console.log(`${colors.cyan}You can now run:${colors.reset}`);
  console.log(`${colors.bright}bun run dev${colors.reset}`);
}

main().catch(error => {
  console.error(`${colors.red}An unexpected error occurred:${colors.reset}`);
  console.error(error);
  process.exit(1);
});
