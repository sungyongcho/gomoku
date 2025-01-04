export const useAlertStore = defineStore("alert", () => {
  const alert = ref<{
    header: string;
    message: string;
    type: "Info" | "Warn";
  }>();

  const doAlert = (header: string, message: string, type: "Info" | "Warn") => {
    alert.value = {
      header,
      message,
      type,
    };
  };

  return {
    alert,
    doAlert,
  };
});
