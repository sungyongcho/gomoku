<script setup lang="ts">
const { docLinks } = storeToRefs(useDocsStore());
</script>
<template>
  <aside class="h-screen border-r-2 border-gray-300 pt-[60px]">
    <Menu :model="docLinks" class="!border-none">
      <template #submenulabel="{ item }">
        <span class="font-bold text-black">{{ item.label }}</span>
      </template>
      <template #item="{ item, props }">
        <NuxtLink
          v-ripple
          class="flex items-center !text-gray-600"
          v-bind="props.action"
          :to="item.url"
        >
          <span :class="item.icon" />
          <span>{{ item.label }}</span>
        </NuxtLink>
      </template>
    </Menu>
  </aside>
</template>
