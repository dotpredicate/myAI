// The server is a separate Python process: reuse whatever answers /api/info on
// the local port, otherwise start `myai-server` detached.

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const DEFAULT_PORT = 8000;
const POLL_INTERVAL_MS = 50;
const START_TIMEOUT_MS = 15_000;
const PROBE_TIMEOUT_MS = 1_000;
const REPO_ROOT = resolve(import.meta.dirname, '../../..');

function localUrl(): string {
  return `http://127.0.0.1:${process.env.MYAI_PORT?.trim() || DEFAULT_PORT}`;
}

async function healthy(url: string): Promise<boolean> {
  try {
    const response = await fetch(`${url}/api/info`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!response.ok) return false;
    return serverPid(await response.json()) !== undefined;
  } catch {
    return false;
  }
}

function serverCommand(): string {
  const local = join(REPO_ROOT, 'myai-server');
  return existsSync(local) ? local : 'myai-server';
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((done) => setTimeout(done, milliseconds));
}

function serverPid(value: unknown): number | undefined {
  if (typeof value !== 'object' || value === null || !('pid' in value)) return undefined;
  return typeof value.pid === 'number' ? value.pid : undefined;
}

/** Return the server base URL, starting `myai-server` in the background if needed. */
export async function resolveServer(): Promise<string> {
  const explicit = process.env.MYAI_URL?.trim();
  if (explicit) return explicit.replace(/\/+$/, '');

  const url = localUrl();
  if (await healthy(url)) return url;

  process.stderr.write('starting myAI server…\n');
  const state = { exited: false };
  const child = spawn(serverCommand(), [], {
    cwd: REPO_ROOT,
    detached: true,
    stdio: 'ignore',
    env: process.env,
  });
  child.once('error', () => {
    state.exited = true;
  });
  child.once('exit', () => {
    state.exited = true;
  });
  child.unref();

  const deadline = Date.now() + START_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await healthy(url)) return url;
    if (state.exited) {
      if (await healthy(url)) return url;
      throw new Error('myai-server exited during startup; run `myai-server` to see why');
    }
    await delay(POLL_INTERVAL_MS);
  }
  throw new Error(`myai-server did not start within ${START_TIMEOUT_MS / 1000}s; run \`myai-server\` to see why`);
}
