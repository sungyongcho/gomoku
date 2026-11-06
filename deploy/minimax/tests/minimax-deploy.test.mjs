import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';

const script = name => new URL(`../${name}`, import.meta.url).pathname;
let root;
let env;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'minimax-deploy-test-'));
  const bin = join(root, 'bin');
  await mkdir(bin);
  const dotenv = join(root, '.env');
  await writeFile(dotenv, [
    'DEPLOY_MINIMAX_HOST=192.0.2.1', 'DEPLOY_MINIMAX_SSH_USER=fixture-user', 'DEPLOY_MINIMAX_SSH_KEY=/fixture/key',
    'DEPLOY_MINIMAX_PORT=8123',
  ].join('\n') + '\n');
  const stub = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
fs.appendFileSync(process.env.COMMAND_LOG, JSON.stringify([path.basename(process.argv[1]), ...process.argv.slice(2)]) + '\\n');
console.log('/tmp/fixture-stage');
`;
  for (const command of ['ssh', 'scp', 'rsync']) {
    await writeFile(join(bin, command), stub, { mode: 0o755 });
  }
  env = { ...process.env, PATH: `${bin}:${process.env.PATH}`, DOTENV_PATH: dotenv, COMMAND_LOG: join(root, 'commands.jsonl') };
});

afterEach(() => rm(root, { recursive: true, force: true }));

async function run(name) {
  const result = spawnSync('bash', [script(name)], { env, encoding: 'utf8', timeout: 15000 });
  assert.equal(result.status, 0, result.stderr);
  return (await readFile(env.COMMAND_LOG, 'utf8')).trim().split('\n').map(JSON.parse);
}

test('host setup reaches the host named by DEPLOY_MINIMAX_* with its key', async () => {
  const calls = await run('01_host_setup.sh');
  const ssh = calls.filter(call => call[0] === 'ssh');
  assert.ok(ssh.length > 0);
  for (const call of ssh) {
    assert.ok(call.includes('fixture-user@192.0.2.1'));
    assert.ok(call.includes('/fixture/key'));
  }
});

test('deploy syncs to the default build directory and publishes the configured port', async () => {
  const calls = await run('02_deploy_minimax.sh');
  const rsync = calls.find(call => call[0] === 'rsync');
  assert.equal(rsync.at(-1), 'fixture-user@192.0.2.1:/home/fixture-user/build/gomoku-minimax/src/');
  assert.ok(calls.some(call => call[0] === 'ssh' && call.join(' ').includes('MINIMAX_PORT=%s') && call.join(' ').includes("'8123'")));
});

test('security list setup needs no host yet and asks for the OCID', async () => {
  const dotenv = join(root, 'no-host.env');
  await writeFile(dotenv, '');
  for (const command of ['oci', 'jq', 'curl']) {
    await writeFile(join(root, 'bin', command), '#!/bin/sh\n', { mode: 0o755 });
  }
  const result = spawnSync('bash', [script('00_security_list.sh')], {
    env: { ...env, DOTENV_PATH: dotenv }, encoding: 'utf8', timeout: 15000,
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /DEPLOY_MINIMAX_SECURITY_LIST_OCID is required/);
});
