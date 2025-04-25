<script setup lang="ts">
const { showSettings } = storeToRefs(useGameStore());
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
    <AlertModal />
    <SettingModal v-model:visible="showSettings" />
  </NuxtLayout>
</template>
