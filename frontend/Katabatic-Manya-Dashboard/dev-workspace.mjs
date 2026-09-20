import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// One Vite process imports the four existing page implementations.
const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const host = resolve(frontend, 'results-dashboard');
const cli = resolve(host, 'node_modules/vite/bin/vite.js');
if (!existsSync(cli)) {
  console.error('Install dependencies: npm --prefix frontend/results-dashboard install');
  process.exit(1);
}

const child = spawn(process.execPath, [cli, '--port', '5173', '--strictPort', '--clearScreen', 'false'], {
  cwd: host,
  stdio: 'inherit',
});
let stopping = false;
function stop() {
  stopping = true;
  child.kill('SIGTERM');
}
process.once('SIGINT', stop);
process.once('SIGTERM', stop);
child.once('error', (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
child.once('exit', (code) => {
  process.exitCode = stopping ? 0 : (code ?? 1);
});
