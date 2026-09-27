import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

test('Gomoku deploy does not change DocReview DNS, bindings or routes even with old env values', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gomoku-deploy-test-'));
  try {
    const bin = join(root, 'bin');
    await mkdir(bin);
    const log = join(root, 'commands.jsonl');
    const dotenv = join(root, '.env');
    await writeFile(dotenv, [
      'DEPLOY_GCP_PROJECT=fixture', 'DEPLOY_GCP_REGION=fixture', 'DEPLOY_GCP_ZONE=fixture',
      'DEPLOY_GCP_REPO=fixture', 'DEPLOY_MINIMAX_VM=fixture', 'DEPLOY_ALPHAZERO_VM=fixture',
      'DEPLOY_MINIMAX_MACHINE=fixture', 'DEPLOY_ALPHAZERO_MACHINE=fixture',
      'DEPLOY_DOMAIN=example.test', 'DEPLOY_SA_NAME=fixture', 'DEPLOY_USER_EMAIL=fixture@example.test',
      'DEPLOY_MINIMAX_IP=192.0.2.1', 'DEPLOY_ALPHAZERO_IP=192.0.2.2',
      'CLOUDFLARE_ACCOUNT_ID=fixture-account', 'CLOUDFLARE_API_TOKEN=fixture-token',
      'DEPLOY_DOCREVIEW_IP=192.0.2.3', 'DEPLOY_DOCREVIEW_ORIGIN=http://docreview.invalid:8880',
      'DEPLOY_DOCREVIEW_SITE_ORIGIN=https://docreview.invalid',
    ].join('\n') + '\n');
    const stub = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
fs.appendFileSync(process.env.COMMAND_LOG, JSON.stringify([path.basename(process.argv[1]), ...process.argv.slice(2)]) + '\\n');
console.log(JSON.stringify({ success: true, result: [{ id: 'fixture-record' }], status: 'ok' }));
`;
    for (const command of ['curl', 'npx']) {
      await writeFile(join(bin, command), stub, { mode: 0o755 });
    }
    const result = spawnSync('bash', [new URL('../03_deploy_cloudflare.sh', import.meta.url).pathname], {
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, DOTENV_PATH: dotenv, COMMAND_LOG: log },
      encoding: 'utf8', timeout: 15000,
    });
    assert.equal(result.status, 0, result.stderr);
    const calls = (await readFile(log, 'utf8')).trim().split('\n').map(JSON.parse);
    const dnsWrites = calls.filter(args => args[0] === 'curl' && args.includes('-d'));
    assert.deepEqual(dnsWrites.map(args => JSON.parse(args[args.indexOf('-d') + 1]).name), [
      'minimax-api.example.test', 'alphazero-api.example.test',
    ]);
    const deploy = calls.find(args => args[0] === 'npx' && args[1] === 'wrangler');
    assert.ok(deploy);
    assert.deepEqual(deploy.filter(value => /_ORIGIN:/.test(value)), [
      'MINIMAX_ORIGIN:http://minimax-api.example.test:8080',
      'ALPHAZERO_ORIGIN:http://alphazero-api.example.test:8080',
    ]);
    assert.ok(!calls.some(args => args.join(' ').toLowerCase().includes('docreview')));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
