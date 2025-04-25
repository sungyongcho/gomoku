export const useAlertStore = defineStore("alert", () => {
  const alert = ref<{
    header: string;
    message: string;
    type: "Info" | "Warn";
    actionLabel?: string;
    action?: () => void;
    actionIcon?: string;
  }>();

  const doAlert = ({
    header,
    message,
    type,
    actionLabel,
    action,
    actionIcon,
  }: {
    header: string;
    message: string;
    type: "Info" | "Warn";
    actionLabel?: string;
    actionIcon?: string;
    action?: () => void;
  }) => {
    alert.value = {
      header,
      message,
      type,
      actionLabel,
      action,
      actionIcon: actionIcon || "pi pi-times",
    };
  };

  const closeAlert = () => {
    alert.value = undefined;
  };

  return {
    alert,
    doAlert,
    closeAlert,
  };
});
