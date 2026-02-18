const HEALTHY_PROBE_INTERVAL_MS = 30000;
const DEGRADED_PROBE_INTERVAL_MS = 5000;

export default defineNuxtPlugin(() => {
  const { check } = useBackendHealth();
  const maintenance = useMaintenanceStore();
  const { settings } = storeToRefs(useGameStore());

  let probeTimer: ReturnType<typeof setTimeout> | null = null;
  let probing = false;

  const clearProbeTimer = () => {
    if (!probeTimer) return;
    clearTimeout(probeTimer);
    probeTimer = null;
  };

  const scheduleNextProbe = (isHealthy: boolean) => {
    clearProbeTimer();
    probeTimer = setTimeout(
      runProbe,
      isHealthy ? HEALTHY_PROBE_INTERVAL_MS : DEGRADED_PROBE_INTERVAL_MS,
    );
  };

  const runProbe = async () => {
    if (probing) return;
    probing = true;
    try {
      const ok = await check(settings.value.ai);
      if (ok) {
        maintenance.reportBackendSuccess();
      } else {
        maintenance.reportBackendFailure();
      }
      scheduleNextProbe(ok);
    } finally {
      probing = false;
    }
  };

  watch(
    () => settings.value.ai,
    () => {
      maintenance.setMaintenance(false);
      clearProbeTimer();
      void runProbe();
    },
    { immediate: true },
  );

  window.addEventListener("beforeunload", clearProbeTimer, { once: true });
});
