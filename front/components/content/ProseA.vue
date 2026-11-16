<template>
  <NuxtLink
    :href="props.href"
    :target="props.target"
    :prefetch="!isAbsoluteUrl"
    class="text-gray-500 underline hover:text-black"
  >
    <slot />
  </NuxtLink>
</template>

<script setup lang="ts">
import type { PropType } from "vue";

const props = defineProps({
  href: {
    type: String,
    default: "",
  },
  target: {
    type: String as PropType<
      | "_blank"
      | "_parent"
      | "_self"
      | "_top"
      | (string & object)
      | null
      | undefined
    >,
    default: undefined,
    required: false,
  },
});

// Nuxt prefetches payloads for links on the current hostname and rejects full URLs,
// which logs NUXT_E7001 for links like https://sungyongcho.com on that domain.
const isAbsoluteUrl = computed(() =>
  /^([a-z][a-z\d+.-]*:|\/\/)/i.test(props.href),
);
</script>
