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
  root = await mkdtemp(join(tmpdir(), 'alphazero-deploy-test-'));
  const bin = join(root, 'bin');
  await mkdir(bin);
  const dotenv = join(root, '.env');
  // Deliberately has no Minimax or Cloudflare values: these scripts must not need them.
  await writeFile(dotenv, [
    'DEPLOY_GCP_PROJECT=fixture-project', 'DEPLOY_GCP_REGION=fixture-region', 'DEPLOY_GCP_ZONE=fixture-zone',
    'DEPLOY_GCP_REPO=fixture-repo', 'DEPLOY_ALPHAZERO_VM=gomoku-alphazero', 'DEPLOY_ALPHAZERO_MACHINE=fixture-machine',
    'DEPLOY_SA_NAME=fixture-sa', 'DEPLOY_USER_EMAIL=fixture@example.test', 'ALPHAZERO_INFER_BACKEND=onnx-int8',
  ].join('\n') + '\n');
  const ckpt = join(root, 'champion.pt');
  await writeFile(ckpt, 'fixture');
  // `describe` fails so setup takes its create paths, except the IP lookup it needs to print.
  const stub = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const args = process.argv.slice(2);
fs.appendFileSync(process.env.COMMAND_LOG, JSON.stringify([path.basename(process.argv[1]), ...args]) + '\\n');
if (args.includes('--format=get(address)')) { console.log('192.0.2.9'); process.exit(0); }
if (args.includes('describe')) process.exit(1);
if (path.basename(process.argv[1]) === 'curl') console.log('198.51.100.0/24');
`;
  for (const command of ['gcloud', 'docker', 'curl', 'sleep']) {
    await writeFile(join(bin, command), stub, { mode: 0o755 });
  }
  env = {
    ...process.env,
    PATH: `${bin}:${process.env.PATH}`,
    DOTENV_PATH: dotenv,
    COMMAND_LOG: join(root, 'commands.jsonl'),
    CKPT_SOURCE: ckpt,
    CKPT_DEST: join(root, 'dest.pt'),
    IMAGE_TAG: 'fixture-tag',
  };
});

afterEach(() => rm(root, { recursive: true, force: true }));

async function run(name, ...args) {
  const result = spawnSync('bash', [script(name), ...args], { env, encoding: 'utf8', timeout: 15000 });
  assert.equal(result.status, 0, result.stderr);
  const calls = (await readFile(env.COMMAND_LOG, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.ok(!calls.some(call => call.join(' ').toLowerCase().includes('minimax')), 'touched Minimax');
  return { calls, stdout: result.stdout };
}

test('setup creates one AlphaZero VM and IP and prints the IP for .env', async () => {
  const { calls, stdout } = await run('01_setup.sh');
  const created = sub => calls.filter(c => c[0] === 'gcloud' && c.slice(1, 3).join(' ') === `compute ${sub}` && c[3] === 'create').map(c => c[4]);
  assert.deepEqual(created('instances'), ['gomoku-alphazero']);
  assert.deepEqual(created('addresses'), ['gomoku-alphazero-ip']);
  assert.match(stdout, /DEPLOY_ALPHAZERO_IP=192\.0\.2\.9/);
});

test('deploy builds one image and restarts only the AlphaZero VM', async () => {
  const { calls } = await run('02_deploy.sh');
  const builds = calls.filter(c => c[0] === 'docker' && c[1] === 'buildx');
  assert.equal(builds.length, 1);
  assert.ok(builds[0].includes('fixture-region-docker.pkg.dev/fixture-project/fixture-repo/gomoku-alphazero:fixture-tag'));
  assert.ok(builds[0].at(-1).endsWith('/alphazero'));
  assert.ok(builds[0].some(arg => arg.endsWith('/alphazero/infra/image/Dockerfile.prod')));
  const resets = calls.filter(c => c[0] === 'gcloud' && c[3] === 'reset');
  assert.deepEqual(resets.map(c => c[4]), ['gomoku-alphazero']);
});

test('teardown removes only AlphaZero resources', async () => {
  const { calls } = await run('teardown.sh', '--yes');
  const deletedVms = calls.filter(c => c[0] === 'gcloud' && c[2] === 'instances' && c[3] === 'delete').map(c => c[4]);
  assert.deepEqual(deletedVms, ['gomoku-alphazero']);
});
