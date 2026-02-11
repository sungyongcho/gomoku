export const useEnv = () => {
  const { settings } = storeToRefs(useGameStore());
  const { FRONT_WHERE, LOCAL_MINIMAX, LOCAL_ALPHAZERO } =
    useRuntimeConfig().public;

  const getSocketUrl = () => {
    if (FRONT_WHERE === "local") {
      const port =
        settings.value.ai === "minimax" ? LOCAL_MINIMAX : LOCAL_ALPHAZERO;
      return `ws://localhost:${port}/ws`;
    }
    return settings.value.ai === "minimax"
      ? `wss://sungyongcho.com/minimax/ws`
      : `wss://sungyongcho.com/alphazero/ws`;
  };

  const getDebugSocketUrl = () => {
    if (FRONT_WHERE === "local") {
      const port =
        settings.value.ai === "minimax" ? LOCAL_MINIMAX : LOCAL_ALPHAZERO;
      return `ws://localhost:${port}/ws/debug`;
    }

    return settings.value.ai === "minimax"
      ? `wss://sungyongcho.com/minimax/ws/debug`
      : `wss://sungyongcho.com/alphazero/ws/debug`;
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
