<script setup lang="ts">
const props = defineProps({
  image: {
    type: String,
    default: "",
  },
  loading: {
    type: Boolean,
    default: false,
  },
  size: {
    type: String as PropType<"normal" | "xlarge" | undefined>,
    default: "normal",
  },
});
</script>
<template>
  <div class="relative">
    <Avatar
      class="border-4 border-gray-100 shadow-[0_0_4px_1px_black]"
      :size="size"
      shape="circle"
      :image="image"
    />
    <div
      class="absolute left-0 top-0 flex h-full w-full items-center justify-center rounded-[50%] bg-black bg-opacity-50"
      v-if="loading"
    >
      <i class="pi pi-spin pi-spinner text-2xl text-white"></i>
    </div>
  </div>
</template>
