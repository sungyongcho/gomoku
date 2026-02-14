export const useMaintenanceStore = defineStore("maintenance", () => {
  const showMaintenance = ref(false);

  const setMaintenance = (value: boolean) => {
    showMaintenance.value = value;
  };

  return {
    showMaintenance,
    setMaintenance,
  };
});
