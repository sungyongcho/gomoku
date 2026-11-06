import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { afterEach, mock, test } from 'node:test';
import worker from '../cloudflare-worker.js';

const env = {
  ALPHAZERO_ORIGIN: 'http://alpha.example:8080',
  MINIMAX_ORIGIN: 'http://mini.example:8080',
};
afterEach(() => mock.restoreAll());

for (const [path, target] of [
  ['/alphazero/ws?match=1', 'http://alpha.example:8080/ws?match=1'],
  ['/minimax/ws', 'http://mini.example:8080/ws'],
  ['/gomoku', 'https://omoku.netlify.app/gomoku'],
  ['/gomoku/assets/game.js', 'https://omoku.netlify.app/gomoku/assets/game.js'],
]) {
  test(`preserves Gomoku proxy destination for ${path}`, async () => {
    const upstream = new Response('upstream', { status: 200 });
    const fetch = mock.method(globalThis, 'fetch', async (url, init) => {
      assert.equal(url, target);
      assert.equal(init.headers.get('host'), new URL(target).host);
      assert.equal(init.redirect, 'manual');
      assert.equal(init.headers.get('upgrade'), 'websocket');
      return upstream;
    });
    const result = await worker.fetch(new Request(`https://sungyongcho.com${path}`, {
      headers: { Upgrade: 'websocket' },
    }), env);
    assert.equal(result, upstream);
    assert.equal(fetch.mock.callCount(), 1);
  });
}

test('keeps Gomoku redirects on the public domain', async () => {
  mock.method(globalThis, 'fetch', async () => new Response(null, {
    status: 301, headers: { Location: 'https://omoku.netlify.app/gomoku/?lang=ko' },
  }));
  const result = await worker.fetch(new Request('https://sungyongcho.com/gomoku'), env);
  assert.equal(result.status, 301);
  assert.equal(result.headers.get('location'), 'https://sungyongcho.com/gomoku/?lang=ko');
});

test('future Gomoku deployments own only game routes', async () => {
  const config = await readFile(new URL('../wrangler.toml', import.meta.url), 'utf8');
  const routes = [...config.matchAll(/pattern\s*=\s*"([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(routes, [
    'sungyongcho.com/alphazero/*', 'sungyongcho.com/minimax/*',
    'sungyongcho.com/gomoku', 'sungyongcho.com/gomoku/*',
  ]);
});
