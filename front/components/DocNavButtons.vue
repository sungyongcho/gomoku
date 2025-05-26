<script setup lang="ts">
const props = defineProps({
  prev: {
    type: Object as PropType<{ path: string; title: string }>,
    default: null,
  },
  next: {
    type: Object as PropType<{ path: string; title: string }>,
    default: null,
  },
});
</script>
<template>
  <div class="w-full px-6">
    <div class="flex justify-between py-6 gap-4 border-t-[1px] border-gray-200">
      <Button v-if="prev" size="small">
        <NuxtLink :to="prev" class="flex items-center gap-2">
          <i class="pi pi-arrow-left text-sm"></i> Prev
        </NuxtLink>
      </Button>
      <span v-else></span>

      <Button v-if="props.next" size="small">
        <NuxtLink :to="next" class="flex items-center gap-2">
          Next <i class="pi pi-arrow-right text-sm"></i>
        </NuxtLink>
      </Button>
      <span v-else></span>
    </div>
  </div>
</template>
