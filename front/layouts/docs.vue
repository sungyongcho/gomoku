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
  <div class="flex h-screen flex-col">
    <header class="fixed top-0 left-0 right-0 z-[11]">
      <MHeader doc-page class="w-full" />
    </header>
    <div class="flex-1 min-h-0 overflow-auto pt-14">
      <slot />
    </div>
    <MFooter class="w-full shrink-0" />
  </div>
</template>
