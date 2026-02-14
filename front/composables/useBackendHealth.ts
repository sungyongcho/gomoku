const WS_OPEN_TIMEOUT_MS = 4000;

function checkWsReachable(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    if (typeof WebSocket === "undefined") {
      resolve(false);
      return;
    }
    const timeout = setTimeout(() => {
      try {
        ws.close();
      } catch {
      }
      resolve(false);
    }, WS_OPEN_TIMEOUT_MS);
    let resolved = false;
    const done = (ok: boolean) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      try {
        ws.close();
      } catch {
      }
      resolve(ok);
    };
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      clearTimeout(timeout);
      resolve(false);
      return;
    }
    ws.onopen = () => done(true);
    ws.onerror = () => done(false);
    ws.onclose = () => done(false);
  });
}

/**
 * Returns backend WebSocket URLs to health-check (minimax and alphazero).
 * Uses runtime config so it works in both local and prod.
 */
function getBackendWsUrls(): string[] {
  const config = useRuntimeConfig().public;
  const where = config.FRONT_WHERE as string | undefined;
  if (where === "local") {
    const minimaxPort = config.LOCAL_MINIMAX as string | number | undefined;
    const alphazeroPort = config.LOCAL_ALPHAZERO as string | number | undefined;
    const urls: string[] = [];
    if (minimaxPort != null) urls.push(`ws://localhost:${minimaxPort}/ws`);
    if (alphazeroPort != null) urls.push(`ws://localhost:${alphazeroPort}/ws`);
    return urls;
  }
  return [
    "wss://sungyongcho.com/minimax/ws",
    "wss://sungyongcho.com/alphazero/ws",
  ];
}

/**
 * Checks that all backends (minimax and alphazero) are reachable via WebSocket.
 * Resolves to true only if every backend opens within the timeout.
 */
export function useBackendHealth() {
  const check = async (): Promise<boolean> => {
    const urls = getBackendWsUrls();
    if (urls.length === 0) return true;
    const results = await Promise.all(urls.map((url) => checkWsReachable(url)));
    return results.every(Boolean);
  };

  return { check, getBackendWsUrls };
}
