export const useEnv = () => {
  const { FRONT_WHERE } = useRuntimeConfig().public;
  const { settings } = storeToRefs(useGameStore());
  const { LOCAL_MINIMAX, LOCAL_ALPHAZERO } = useRuntimeConfig().public;

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

  return {
    getSocketUrl,
    getDebugSocketUrl,
  };
};
