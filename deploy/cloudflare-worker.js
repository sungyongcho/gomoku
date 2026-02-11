/**
 * Cloudflare Worker: path-based routing for Gomoku backends.
 *
 * Expected env vars (set via wrangler.toml):
 * - ALPHAZERO_ORIGIN: e.g. "http://203.0.113.10:8080"
 * - MINIMAX_ORIGIN:   e.g. "http://203.0.113.11:8080"
 *
 * Routes:
 * - /alphazero/* -> ALPHAZERO_ORIGIN/*
 * - /minimax/*   -> MINIMAX_ORIGIN/*
 *
 * Note: This Worker is intended to be deployed on narrow routes
 * (e.g. sungyongcho.com/alphazero/* and sungyongcho.com/minimax/*).
 */

function _matchesPrefix(pathname, prefix) {
  return pathname === prefix || pathname.startsWith(prefix + "/");
}

function _joinPaths(basePath, suffixPath) {
  const base = basePath && basePath !== "/" ? basePath.replace(/\/+$/, "") : "";
  const suffix = suffixPath.startsWith("/") ? suffixPath : "/" + suffixPath;
  return (base || "") + suffix;
}

function _proxyToOrigin(request, origin, stripPrefix) {
  if (!origin) {
    return new Response("Origin is not configured", { status: 500 });
  }

  const originUrl = new URL(origin);
  const url = new URL(request.url);

  let newPath = url.pathname.slice(stripPrefix.length);
  if (!newPath) newPath = "/";
  url.protocol = originUrl.protocol;
  url.hostname = originUrl.hostname;
  url.port = originUrl.port;
  url.pathname = _joinPaths(originUrl.pathname, newPath);

  // Preserve method/headers/body. Overwrite Host so origin sees the expected host:port.
  const headers = new Headers(request.headers);
  headers.set("Host", originUrl.host);

  const init = {
    method: request.method,
    headers,
    redirect: "manual",
    body: request.body,
  };

  return fetch(url.toString(), init);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (_matchesPrefix(url.pathname, "/alphazero")) {
      return _proxyToOrigin(request, env.ALPHAZERO_ORIGIN, "/alphazero");
    }
    if (_matchesPrefix(url.pathname, "/minimax")) {
      return _proxyToOrigin(request, env.MINIMAX_ORIGIN, "/minimax");
    }

    // If the Worker is only deployed on /alphazero/* and /minimax/*,
    // this branch is never hit. Kept for safety.
    return fetch(request);
  },
};

