<script setup lang="ts">
import type { PropType } from "vue";

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
  color: {
    type: String as PropType<"black" | "white" | undefined>,
    default: "black",
  },
  active: {
    type: Boolean,
    default: false,
  },
});
</script>
<template>
  <div
    class="relative flex items-center justify-center border-[6px] border-transparent -lg:border-[3px]"
    :class="{ ['rounded-[50%] border-yellow-500']: active }"
  >
    <Avatar
      class="border-4"
      :size="size"
      shape="circle"
      :image="image"
      :class="{
        ['border-black shadow-[0_0_2px_1px_white]']: color === 'black',
        ['border-white shadow-[0_0_2px_1px_black]']: color === 'white',
      }"
    />
    <div
      class="absolute left-0 top-0 flex h-full w-full items-center justify-center rounded-[50%] bg-black bg-opacity-50"
      v-if="loading"
    >
      <i class="pi pi-spin pi-spinner text-2xl text-white"></i>
    </div>
  </div>
</template>
