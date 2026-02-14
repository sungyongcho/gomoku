<script setup lang="ts">
const { showSettings } = storeToRefs(useGameStore());
const { fetchDocLinks } = useDocsStore();

await callOnce(fetchDocLinks);
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
    <AlertModal />
    <MaintenanceModal />
    <SettingModal v-model:visible="showSettings" />
  </NuxtLayout>
</template>
