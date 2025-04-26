#!/usr/bin/env node

/**
 * This script updates all dependencies in package.json to their latest versions
 * It uses the npm registry to check for the latest versions
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

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

// Read package.json
const packageJsonPath = path.join(process.cwd(), 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

// Function to fetch latest version from npm registry
function getLatestVersion(packageName) {
  return new Promise((resolve, reject) => {
    const url = `https://registry.npmjs.org/${packageName}`;

    https.get(url, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          if (res.statusCode === 404) {
            resolve({ name: packageName, latest: null, error: 'Package not found' });
            return;
          }

          const packageData = JSON.parse(data);
          const latestVersion = packageData['dist-tags']?.latest;

          resolve({ name: packageName, latest: latestVersion });
        } catch (error) {
          resolve({ name: packageName, latest: null, error: error.message });
        }
      });
    }).on('error', (error) => {
      resolve({ name: packageName, latest: null, error: error.message });
    });
  });
}

// Known peer dependency constraints
const PEER_DEPENDENCY_CONSTRAINTS = {
  '@builder.io/qwik': {
    'vite': '^5.0.0'
  },
  '@builder.io/qwik-city': {
    'vite': '^5.0.0'
  }
};

// Update dependencies in package.json
async function updateDependencies() {
  console.log(`${colors.bright}${colors.magenta}Updating dependencies in package.json${colors.reset}\n`);

  const dependencyTypes = [
    { name: 'dependencies', deps: packageJson.dependencies || {} },
    { name: 'devDependencies', deps: packageJson.devDependencies || {} }
  ];

  let hasChanges = false;
  let hasErrors = false;
  let peerDependencyConstraints = {};

  // First pass: collect all peer dependency constraints
  for (const { name, deps } of dependencyTypes) {
    const packages = Object.keys(deps);

    for (const pkg of packages) {
      if (PEER_DEPENDENCY_CONSTRAINTS[pkg]) {
        Object.assign(peerDependencyConstraints, PEER_DEPENDENCY_CONSTRAINTS[pkg]);
      }
    }
  }

  // Second pass: update versions respecting peer dependency constraints
  for (const { name, deps } of dependencyTypes) {
    console.log(`${colors.bright}Checking ${name}:${colors.reset}`);

    const packages = Object.keys(deps);
    const results = await Promise.all(packages.map(getLatestVersion));

    for (const result of results) {
      const currentVersion = deps[result.name];

      if (result.error) {
        console.log(`  ${colors.red}✗ ${result.name}: ${result.error}${colors.reset}`);
        hasErrors = true;
        continue;
      }

      if (!result.latest) {
        console.log(`  ${colors.yellow}? ${result.name}: Could not determine latest version${colors.reset}`);
        continue;
      }

      // Extract the current version without the range specifier
      const currentVersionNumber = currentVersion.replace(/^[~^]/, '');

      // Check if this package has peer dependency constraints
      if (peerDependencyConstraints[result.name]) {
        const constraint = peerDependencyConstraints[result.name];
        console.log(`  ${colors.yellow}! ${result.name}: Has peer dependency constraint: ${constraint}${colors.reset}`);

        // Keep the current version to respect peer dependencies
        console.log(`  ${colors.dim}• ${result.name}: ${currentVersion} (keeping due to peer dependency constraints)${colors.reset}`);
        continue;
      }

      if (currentVersionNumber !== result.latest) {
        // Keep the same range specifier (^ or ~)
        const rangeSpecifier = currentVersion.startsWith('^') ? '^' :
                              currentVersion.startsWith('~') ? '~' : '';

        // Update the version
        deps[result.name] = `${rangeSpecifier}${result.latest}`;

        console.log(`  ${colors.green}✓ ${result.name}: ${currentVersion} → ${deps[result.name]}${colors.reset}`);
        hasChanges = true;
      } else {
        console.log(`  ${colors.dim}• ${result.name}: ${currentVersion} (up to date)${colors.reset}`);
      }
    }

    console.log('');
  }

  if (hasChanges) {
    // Write updated package.json
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
    console.log(`${colors.green}${colors.bright}✓ package.json updated with latest versions${colors.reset}`);
  } else {
    console.log(`${colors.blue}All dependencies are up to date!${colors.reset}`);
  }

  if (hasErrors) {
    console.log(`\n${colors.yellow}⚠ Some packages had errors. You may need to check them manually.${colors.reset}`);
  }

  console.log(`\n${colors.cyan}Run ${colors.bright}bun install${colors.reset}${colors.cyan} to install the updated dependencies.${colors.reset}`);
}

updateDependencies().catch(error => {
  console.error(`${colors.red}An error occurred:${colors.reset}`, error);
});
