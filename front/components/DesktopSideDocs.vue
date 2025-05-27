<script setup lang="ts">
const { docLinks } = storeToRefs(useDocsStore());
</script>
<template>
  <aside class="overflow-y-auto border-r-[1px] border-gray-200 shadow-sm">
    <Menu :model="docLinks" class="!border-none">
      <template #submenulabel="{ item }">
        <span class="font-bold text-black">{{ item.label }}</span>
      </template>
      <template #item="{ item, props }">
        <NuxtLink
          v-ripple
          class="link flex items-center !text-gray-600"
          v-bind="props.action"
          :to="item.url"
        >
          <span :class="`pi ${item.icon}`" />
          <span>{{ item.label }}</span>
        </NuxtLink>
      </template>
    </Menu>
  </aside>
</template>
<style>
.link.router-link-active {
  @apply bg-gray-100 !text-black;
}
</style>
