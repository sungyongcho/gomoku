export const useMaintenanceStore = defineStore("maintenance", () => {
  const showMaintenance = ref(false);
  const consecutiveFailures = ref(0);
  const FAILURE_THRESHOLD = 3;

  const setMaintenance = (value: boolean) => {
    showMaintenance.value = value;
    if (!value) {
      consecutiveFailures.value = 0;
    }
  };

  const reportBackendFailure = () => {
    consecutiveFailures.value += 1;
    if (consecutiveFailures.value >= FAILURE_THRESHOLD) {
      showMaintenance.value = true;
    }
  };

  const reportBackendSuccess = () => {
    consecutiveFailures.value = 0;
    showMaintenance.value = false;
  };

  const forceMaintenance = () => {
    consecutiveFailures.value = FAILURE_THRESHOLD;
    showMaintenance.value = true;
  };

  return {
    showMaintenance,
    consecutiveFailures,
    setMaintenance,
    reportBackendFailure,
    reportBackendSuccess,
    forceMaintenance,
  };
});
