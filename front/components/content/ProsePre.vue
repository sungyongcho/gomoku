<template>
  <div class="relative my-2 rounded-lg bg-slate-800 px-4 py-1 text-gray-100">
    <pre :class="$props.class">
      <slot />
    </pre>
    <span class="absolute bottom-[16px] right-[16px] text-sm !text-gray-100">
      {{ language || filename }}
    </span>
  </div>
</template>

<script setup lang="ts">
defineProps({
  code: {
    type: String,
    default: "",
  },
  language: {
    type: String,
    default: null,
  },
  filename: {
    type: String,
    default: null,
  },
  highlights: {
    type: Array as () => number[],
    default: () => [],
  },
  meta: {
    type: String,
    default: null,
  },
  class: {
    type: String,
    default: null,
  },
});
</script>

<style>
pre code .line {
  display: block;
}
</style>
