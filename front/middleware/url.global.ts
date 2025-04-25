export default defineNuxtRouteMiddleware((to, from) => {
  if (to.query.data) {
    // store data
    const { importData } = useGameStore();
    const { doAlert } = useAlertStore();
    try {
      importData(to.query.data as string);
    } catch (e) {
      doAlert({
        header: "Import Data Error",
        message: "Invalid data",
        type: "Warn",
      });
    }

    return navigateTo({
      path: to.path,
      query: {},
    });
  }
});
