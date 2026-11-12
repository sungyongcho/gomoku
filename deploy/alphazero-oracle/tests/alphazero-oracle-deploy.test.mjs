import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';

const script = name => new URL(`../${name}`, import.meta.url).pathname;
let root;
let env;
let dotenvLines;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'alphazero-oracle-deploy-test-'));
  const bin = join(root, 'bin');
  await mkdir(bin);
  const checkpoint = join(root, 'model.pt');
  await writeFile(checkpoint, 'fixture');
  dotenvLines = [
    'DEPLOY_ALPHAZERO_A1_HOST=192.0.2.7', 'DEPLOY_ALPHAZERO_A1_SSH_USER=fixture-user',
    'DEPLOY_ALPHAZERO_A1_SSH_KEY=/fixture/key', 'DEPLOY_ALPHAZERO_A1_PORT=8123',
    `DEPLOY_ALPHAZERO_A1_CHECKPOINT=${checkpoint}`,
  ];
  const stub = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
fs.appendFileSync(process.env.COMMAND_LOG, JSON.stringify([path.basename(process.argv[1]), ...process.argv.slice(2)]) + '\\n');
console.log('/tmp/fixture-stage');
`;
  for (const command of ['ssh', 'scp', 'rsync']) {
    await writeFile(join(bin, command), stub, { mode: 0o755 });
  }
  env = { ...process.env, PATH: `${bin}:${process.env.PATH}`, COMMAND_LOG: join(root, 'commands.jsonl') };
});

afterEach(() => rm(root, { recursive: true, force: true }));

async function writeDotenv(lines = dotenvLines) {
  const dotenv = join(root, '.env');
  await writeFile(dotenv, lines.join('\n') + '\n');
  env.DOTENV_PATH = dotenv;
}

function runRaw(name, extraEnv = {}) {
  return spawnSync('bash', [script(name)], { env: { ...env, ...extraEnv }, encoding: 'utf8', timeout: 15000 });
}

async function run(name, extraEnv = {}) {
  const result = runRaw(name, extraEnv);
  assert.equal(result.status, 0, result.stderr);
  return (await readFile(env.COMMAND_LOG, 'utf8')).trim().split('\n').map(JSON.parse);
}

test('host setup reaches the host named by DEPLOY_ALPHAZERO_A1_* with its key', async () => {
  await writeDotenv();
  const calls = await run('01_host_setup.sh');
  const ssh = calls.filter(call => call[0] === 'ssh');
  const setup = ssh.find(call => call.join(' ').includes('host_setup.sh'));
  assert.match(setup.at(-1), /'fixture-user' '8123'; rm -rf/);
  for (const call of ssh) {
    assert.ok(call.includes('fixture-user@192.0.2.7'));
    assert.ok(call.includes('/fixture/key'));
  }
});

test('deploy syncs the source, sends the one checkpoint and writes the serving settings', async () => {
  await writeDotenv();
  const calls = await run('02_deploy_alphazero.sh');
  const rsync = calls.find(call => call[0] === 'rsync');
  assert.equal(rsync.at(-1), 'fixture-user@192.0.2.7:/home/fixture-user/build/gomoku-alphazero/src/');
  assert.ok(rsync.includes('models/'), 'local models/ must not be synced');
  assert.ok(rsync.includes('runs/'), 'training runs must not be synced');
  const checkpoint = calls.find(call => call[0] === 'scp' && call.at(-1).endsWith('/src/models/champion.pt'));
  assert.ok(checkpoint.at(-2).endsWith('model.pt'));
  const settings = calls.find(call => call[0] === 'ssh' && call.join(' ').includes('ALPHAZERO_CONFIG=%s')).join(' ');
  for (const value of ["'8123'", "'configs/deploy-v5.yaml'", "'onnx-static'", "'2'"]) {
    assert.ok(settings.includes(value), `${value} missing from ${settings}`);
  }
});

test('deploy refuses to run without a checkpoint for the chosen config', async () => {
  await writeDotenv(dotenvLines.filter(line => !line.startsWith('DEPLOY_ALPHAZERO_A1_CHECKPOINT=')));
  const result = runRaw('02_deploy_alphazero.sh');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /DEPLOY_ALPHAZERO_A1_CHECKPOINT is required/);
});

test('deploy refuses a config that is not in the repository', async () => {
  await writeDotenv([...dotenvLines, 'DEPLOY_ALPHAZERO_A1_CONFIG=configs/missing.yaml']);
  const result = runRaw('02_deploy_alphazero.sh');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Config not found/);
});

test('the search benchmark uses the served backend and stops the engine only when asked', async () => {
  await writeDotenv([...dotenvLines, 'DEPLOY_ALPHAZERO_A1_CONFIG=configs/deploy-v6b.yaml']);
  let calls = await run('03_bench_search.sh');
  const bench = calls.find(call => call.join(' ').includes('bench_search')).join(' ');
  assert.match(bench, /ALPHAZERO_INFER_BACKEND='onnx-static'/);
  assert.match(bench, /ALPHAZERO_INFER_THREADS='2'/);
  assert.match(bench, /--config 'configs\/deploy-v6b.yaml'/);
  assert.ok(!calls.some(call => call.join(' ').includes('stop alphazero')));

  calls = await run('03_bench_search.sh', { STOP_ENGINE: 'true' });
  const text = calls.map(call => call.join(' '));
  assert.ok(text.findIndex(line => line.includes('stop alphazero')) < text.findLastIndex(line => line.includes('bench_search')));
  assert.ok(text.at(-1).includes('up -d alphazero'));
});
