<script setup lang="ts">
const props = defineProps({
  modelValue: {
    type: Number,
    required: true,
  },
});
</script>
<template>
  <div class="flex items-center gap-2">
    Bad
    <div
      class="relative h-2 w-[120px] rounded-[80px] bg-gradient-to-r from-blue-500 to-red-500"
    >
      <i
        class="pi pi-sort-down-fill absolute bottom-[50%] translate-x-[-50%] text-black"
        :style="{ left: `${Math.min(modelValue, 100)}%` }"
      ></i>
    </div>
    Good
  </div>
</template>
