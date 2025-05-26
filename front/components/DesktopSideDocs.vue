<script setup lang="ts">
const { docLinks } = storeToRefs(useDocsStore());
</script>
<template>
  <aside class="h-screen border-r-2 border-gray-300 pt-[60px]">
    <Menu :model="docLinks" class="!border-none" />
  </aside>
</template>
