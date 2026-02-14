export default defineNuxtPlugin(async () => {
  const { check } = useBackendHealth();
  const ok = await check();
  const maintenance = useMaintenanceStore();
  maintenance.setMaintenance(!ok);
});
