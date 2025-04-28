// Simple test script for the syllable counter
const { exec } = require('child_process');

// Test haikus
const testHaikus = [
  "Morning sun rises\nDew drops glisten on green leaves\nA new day begins",
  "Typing on keyboard\nThoughts flow into characters\nTasks become haikus",
  "Mountain silhouette\nShadows dance across the lake\nPeace in solitude",
  "Deadline approaching\nFingers race across the keys\nWork becomes a blur",
  "Empty task list waits\nIdeas form in my mind\nTime to write them down"
];

// Run the development server
console.log("Starting development server...");
const server = exec('cd .. && bun run dev');

// Wait for the server to start
setTimeout(() => {
  console.log("Opening browser to test the syllable counter...");
  exec('start http://localhost:5173');
  
  console.log("Test haikus to try:");
  testHaikus.forEach((haiku, index) => {
    console.log(`\nHaiku ${index + 1}:`);
    console.log(haiku);
  });
  
  console.log("\nPress Ctrl+C to stop the server when done testing.");
}, 5000);

// Handle server output
server.stdout.on('data', (data) => {
  console.log(`Server: ${data}`);
});

server.stderr.on('data', (data) => {
  console.error(`Server Error: ${data}`);
});

// Handle process exit
process.on('SIGINT', () => {
  console.log("Stopping server...");
  server.kill();
  process.exit();
});
