export const useEnv = () => {
  const { settings } = storeToRefs(useGameStore());
  const { FRONT_WHERE, LOCAL_MINIMAX, LOCAL_ALPHAZERO } =
    useRuntimeConfig().public;

  const getSocketUrl = () => {
    if (FRONT_WHERE === "local") {
      const port =
        settings.value.ai === "minimax" ? LOCAL_MINIMAX : LOCAL_ALPHAZERO;
      return `ws://${window.location.hostname}:${port}/ws`;
    }
    return settings.value.ai === "minimax"
      ? `wss://minimax.sungyongcho.com/ws`
      : `wss://alphazero.sungyongcho.com/ws`;
  };

  const getDebugSocketUrl = () => {
    if (FRONT_WHERE === "local") {
      const port =
        settings.value.ai === "minimax" ? LOCAL_MINIMAX : LOCAL_ALPHAZERO;
      return `ws://${window.location.hostname}:${port}/ws/debug`;
    }

    return settings.value.ai === "minimax"
      ? `wss://minimax.sungyongcho.com/ws/debug`
      : `wss://alphazero.sungyongcho.com/ws/debug`;
  };

  const isLocal = FRONT_WHERE === "local";
  const isProd = !isLocal;

  return {
    getSocketUrl,
    getDebugSocketUrl,
    isLocal,
    isProd,
  };
};
