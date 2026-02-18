const WS_OPEN_TIMEOUT_MS = 8000;
type BackendAi = "minimax" | "alphazero";

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
    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({ type: "ping" }));
      } catch {
        done(false);
      }
    };
    ws.onmessage = (event) => {
      try {
        const payload =
          typeof event.data === "string" ? JSON.parse(event.data) : undefined;
        done(payload?.type === "pong");
      } catch {
        done(false);
      }
    };
    ws.onerror = () => done(false);
    ws.onclose = () => done(false);
  });
}

/**
 * Returns backend WebSocket URLs to health-check (minimax and alphazero).
 * Uses runtime config so it works in both local and prod.
 */
function getBackendWsUrls(ai?: BackendAi): string[] {
  const config = useRuntimeConfig().public;
  const where = config.FRONT_WHERE as string | undefined;
  const selectedAi = ai;
  if (where === "local") {
    const minimaxPort = config.LOCAL_MINIMAX as string | number | undefined;
    const alphazeroPort = config.LOCAL_ALPHAZERO as string | number | undefined;
    const urls: string[] = [];
    if (selectedAi !== "alphazero" && minimaxPort != null)
      urls.push(`ws://localhost:${minimaxPort}/ws`);
    if (selectedAi !== "minimax" && alphazeroPort != null)
      urls.push(`ws://localhost:${alphazeroPort}/ws`);
    return urls;
  }
  const urls: string[] = [];
  if (selectedAi !== "alphazero") {
    urls.push("wss://sungyongcho.com/minimax/ws");
  }
  if (selectedAi !== "minimax") {
    urls.push("wss://sungyongcho.com/alphazero/ws");
  }
  return urls;
}

/**
 * Checks that all backends (minimax and alphazero) respond to WebSocket ping.
 * Resolves to true only if every backend returns a pong within the timeout.
 */
export function useBackendHealth() {
  const check = async (ai?: BackendAi): Promise<boolean> => {
    const urls = getBackendWsUrls(ai);
    if (urls.length === 0) return true;
    const results = await Promise.all(urls.map((url) => checkWsReachable(url)));
    return results.every(Boolean);
  };

  return { check, getBackendWsUrls };
}
