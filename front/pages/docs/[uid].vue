<script setup lang="ts">
const route = useRoute();
const { data: pageData } = await useAsyncData(route.path, () =>
  queryCollection("docs").path(route.path).first(),
);
const { data: surroundData } = await useAsyncData(
  `${route.path}-surrounded`,
  () => {
    return queryCollectionItemSurroundings("docs", route.path);
  },
);

useSeoMeta({
  title: pageData.value?.title,
  description: pageData.value?.description,
});

definePageMeta({
  layout: "docs",
});
</script>

<template>
  <main class="mx-auto flex max-w-[1140px] gap-8">
    <DesktopSideDocs class="w-fit-content -sm:hidden" />

    <div class="max-w-[800px] min-h-[calc(100vh-60px-104px)] flex-1">
      <ContentRenderer v-if="pageData" :value="pageData" class="p-6" />
      <div v-else>Home not found</div>

      <DocNavButtons
        :prev="surroundData?.[0]"
        :next="surroundData?.[1]"
      />
    </div>
  </main>
</template>
