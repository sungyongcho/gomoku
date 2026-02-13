<script setup lang="ts">
useHead({
  link: [
    {
      rel: "stylesheet",
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
    },
  ],
});
</script>

<template>
  <div class="h-screen">
    <MHeader doc-page class="w-full" />
    <slot></slot>
    <MFooter class="w-full" />
  </div>
</template>
