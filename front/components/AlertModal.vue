<script setup lang="ts">
const { alert } = storeToRefs(useAlertStore());
const _visible = computed({
  get() {
    return !!alert.value;
  },
  set(v) {
    if (!v) {
      alert.value = undefined;
    }
  },
});
</script>

<template>
  <Dialog
    v-model:visible="_visible"
    modal
    dismissableMask
    :header="alert?.header"
    closeOnEscape
  >
    <template #header>
      <div class="flex items-center justify-center gap-2 text-[20px] font-bold">
        <i
          class="pi"
          :class="[
            { 'pi-info-circle text-info': alert?.type === 'Info' },
            { 'pi-exclamation-triangle text-warn': alert?.type === 'Warn' },
          ]"
        />
        <span class="mr-5">{{ alert?.header }}</span>
      </div>
    </template>

    <p>
      {{ alert?.message }}
    </p>
  </Dialog>
</template>

<style scoped>
.text-info {
  @apply text-2xl text-blue-500;
}

.text-warn {
  @apply text-2xl text-yellow-600;
}
</style>
