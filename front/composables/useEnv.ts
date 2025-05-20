export const useEnv = () => {
  const { FRONT_WHERE } = useRuntimeConfig().public;

  const getSocketUrl = () => {
    if (FRONT_WHERE === "local") {
      return `ws://${window.location.hostname}:8005/ws/debug`;
    }
    return `wss://minimax.sungyongcho.com/ws/debug`;
  };

  return {
    getSocketUrl,
  };
};
