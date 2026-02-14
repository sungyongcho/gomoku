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
 * - /gomoku/*    -> proxy to https://omoku.netlify.app/gomoku/*
 *
 * Note: This Worker is intended to be deployed on narrow routes
 * (e.g. sungyongcho.com/alphazero/*, /minimax/*, /gomoku/*).
 */
const GOMOKU_SITE_ORIGIN = "https://omoku.netlify.app";

function _matchesPrefix(pathname, prefix) {
  return pathname === prefix || pathname.startsWith(prefix + "/");
}

function _joinPaths(basePath, suffixPath) {
  const base = basePath && basePath !== "/" ? basePath.replace(/\/+$/, "") : "";
  const suffix = suffixPath.startsWith("/") ? suffixPath : "/" + suffixPath;
  return (base || "") + suffix;
}

async function _proxyToOrigin(request, origin, stripPrefix, publicPrefix = null) {
  if (!origin) {
    return new Response("Origin is not configured", { status: 500 });
  }

  const originUrl = new URL(origin);
  const requestUrl = new URL(request.url);
  const upstreamUrl = new URL(requestUrl.toString());

  let newPath = requestUrl.pathname.slice(stripPrefix.length);
  if (!newPath) newPath = "/";
  upstreamUrl.protocol = originUrl.protocol;
  upstreamUrl.hostname = originUrl.hostname;
  upstreamUrl.port = originUrl.port;
  upstreamUrl.pathname = _joinPaths(originUrl.pathname, newPath);

  // Preserve method/headers/body. Overwrite Host so origin sees the expected host:port.
  const headers = new Headers(request.headers);
  headers.set("Host", originUrl.host);

  const init = {
    method: request.method,
    headers,
    redirect: "manual",
    body: request.body,
  };

  const response = await fetch(upstreamUrl.toString(), init);
  if (!publicPrefix) {
    return response;
  }

  const location = response.headers.get("location");
  if (!location) {
    return response;
  }

  let locationUrl;
  try {
    locationUrl = new URL(location, upstreamUrl.toString());
  } catch {
    return response;
  }

  // Keep browser URL on sungyongcho.com by rewriting upstream redirects.
  if (
    locationUrl.protocol !== originUrl.protocol ||
    locationUrl.hostname !== originUrl.hostname
  ) {
    return response;
  }

  const rewritten = new URL(request.url);
  rewritten.pathname = _joinPaths(publicPrefix, locationUrl.pathname);
  rewritten.search = locationUrl.search;
  rewritten.hash = locationUrl.hash;

  const responseHeaders = new Headers(response.headers);
  responseHeaders.set("location", rewritten.toString());
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
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
    if (_matchesPrefix(url.pathname, "/gomoku")) {
      return _proxyToOrigin(
        request,
        GOMOKU_SITE_ORIGIN,
        "",
        "/"
      );
    }

    // If the Worker is only deployed on /alphazero/*, /minimax/*, and /gomoku/*,
    // this branch is never hit. Kept for safety.
    return fetch(request);
  },
};
