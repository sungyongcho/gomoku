<script setup lang="ts">
definePageMeta({
  layout: "game",
});
</script>
<template>
  <main
    class="flex h-[calc(100vh-68px)] w-full items-center justify-center -sm:h-[calc(100vh-128px)]"
  >
    <div
      class="flex max-w-[1280px] items-center justify-center gap-10 -lg:flex-col-reverse"
    >
      <GoBoard />
      <InfoBoard />
    </div>
  </main>
</template>
