<script setup lang="ts">
const { showMaintenance } = storeToRefs(useMaintenanceStore());
</script>

<template>
  <Dialog
    :visible="showMaintenance"
    modal
    :closable="false"
    :close-on-escape="false"
    :dismissable-mask="false"
    :draggable="false"
    class="maintenance-dialog"
  >
    <template #header>
      <div class="flex items-center justify-center gap-2 text-xl font-bold">
        <i class="pi pi-wrench text-amber-500" />
        <span>Under maintenance</span>
      </div>
    </template>
    <p class="text-center text-gray-600 dark:text-gray-400">
      One or more game backends are temporarily unavailable. Please try again
      later.
    </p>
    <template #footer>
      <span class="text-sm text-gray-500">This message will not close.</span>
    </template>
  </Dialog>
</template>

<style scoped>
.maintenance-dialog :deep(.p-dialog-header-close) {
  display: none;
}
</style>
