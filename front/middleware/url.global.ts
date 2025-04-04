export default defineNuxtRouteMiddleware((to, from) => {
  if (to.query.data) {
    // store data
    const { importData } = useGameStore();
    const { doAlert } = useAlertStore();
    try {
      importData(to.query.data as string);
    } catch (e) {
      doAlert("Import Data Error", "Invalid data", "Warn");
    }

    return navigateTo({
      path: to.path,
      query: {},
    });
  }
});
