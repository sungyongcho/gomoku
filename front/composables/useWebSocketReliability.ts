import { onUnmounted, ref, watch } from "vue";
import type { Ref } from "vue";

import type { RequestType } from "~/types/game";

const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30000;
const RECONNECT_MAX_EXPONENT = 5;
const RECONNECT_JITTER_MS = 500;

const REQUEST_TIMEOUT_MS: Record<RequestType, number> = {
  move: 45000,
  evaluate: 12000,
  test: 45000,
};

type SocketStatus = "OPEN" | "CONNECTING" | "CLOSED";

type UseWebSocketReliabilityParams = {
  status: Ref<string>;
  open: () => void;
  close: () => void;
  onRequestTimedOut: (type: RequestType) => void;
};

export const useWebSocketReliability = ({
  status,
  open,
  close,
  onRequestTimedOut,
}: UseWebSocketReliabilityParams) => {
  const maintenance = useMaintenanceStore();
  const reconnectAttempts = ref(0);
  let isTearingDown = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let requestTimeoutTimer: ReturnType<typeof setTimeout> | null = null;

  const clearReconnectTimer = () => {
    if (!reconnectTimer) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  };

  const clearRequestTimeout = () => {
    if (!requestTimeoutTimer) return;
    clearTimeout(requestTimeoutTimer);
    requestTimeoutTimer = null;
  };

  const reconnectDelayMs = () => {
    const exp = Math.min(reconnectAttempts.value, RECONNECT_MAX_EXPONENT);
    const base = Math.min(RECONNECT_MAX_DELAY_MS, RECONNECT_BASE_DELAY_MS * 2 ** exp);
    const jitter = Math.floor(Math.random() * RECONNECT_JITTER_MS);
    return base + jitter;
  };

  const scheduleReconnect = () => {
    if (reconnectTimer) return;
    if (status.value === "OPEN" || status.value === "CONNECTING") return;

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (status.value !== "OPEN" && status.value !== "CONNECTING") {
        reconnectAttempts.value += 1;
        open();
      }
    }, reconnectDelayMs());
  };

  const ensureOpen = () => {
    if (status.value === "OPEN") return true;
    if (status.value === "CLOSED") {
      maintenance.reportBackendFailure();
      open();
    }
    scheduleReconnect();
    return false;
  };

  const startRequestTimeout = (type: RequestType) => {
    clearRequestTimeout();
    requestTimeoutTimer = setTimeout(() => {
      clearRequestTimeout();
      maintenance.forceMaintenance();
      close();
      scheduleReconnect();
      onRequestTimedOut(type);
    }, REQUEST_TIMEOUT_MS[type]);
  };

  watch(
    status,
    (next, prev) => {
      if (isTearingDown) return;

      const nextStatus = next as SocketStatus;
      const prevStatus = prev as SocketStatus | undefined;

      if (nextStatus === "OPEN") {
        reconnectAttempts.value = 0;
        clearReconnectTimer();
        return;
      }

      if (nextStatus === "CLOSED" && prevStatus !== "CLOSED") {
        maintenance.reportBackendFailure();
      }

      if (nextStatus === "CLOSED") {
        scheduleReconnect();
      }
    },
    { immediate: true },
  );

  onUnmounted(() => {
    isTearingDown = true;
    clearReconnectTimer();
    clearRequestTimeout();
  });

  return {
    ensureOpen,
    scheduleReconnect,
    startRequestTimeout,
    clearRequestTimeout,
  };
};
