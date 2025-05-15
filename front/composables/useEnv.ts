export const useEnv = () => {
  const { WHERE } = useRuntimeConfig().public;

  const getSocketUrl = () => {
    if (WHERE === "local") {
      return `ws://${window.location.hostname}:8005/ws/debug`;
    }
    return `wss://minimax.sungyongcho.com/ws/debug`;
  };

  return {
    getSocketUrl,
  };
};
