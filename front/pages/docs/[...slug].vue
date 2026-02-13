<script setup lang="ts">
const route = useRoute();

// Helper to find doc with case-insensitive path matching
const findDocByPath = async (path: string) => {
  // Try exact path first
  let doc = await queryCollection("docs").path(path).first();
  
  // If not found, try case-insensitive matching
  if (!doc) {
    const allDocs = await queryCollection("docs").all();
    const normalizedPath = path.toLowerCase();
    doc = allDocs.find((d) => d.path.toLowerCase() === normalizedPath);
  }
  
  return doc;
};

const { data: pageData } = await useAsyncData(route.path, () =>
  findDocByPath(route.path),
);

const { data: surroundData } = await useAsyncData(
  `${route.path}-surrounded`,
  async () => {
    // Use the actual doc path for surroundings if we found a case-insensitive match
    const doc = await findDocByPath(route.path);
    const actualPath = doc?.path || route.path;
    return queryCollectionItemSurroundings("docs", actualPath);
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
  <main class="mx-auto flex max-w-[1140px] relative">
    <DesktopSideDocs
      class="sticky top-4 self-start w-[250px] min-h-0 max-h-[calc(100vh-60px-92px)]"
    />

    <div class="max-w-[800px] min-h-[calc(100vh-60px-92px)] flex-1">
      <ContentRenderer v-if="pageData" :value="pageData" class="p-6" />
      <div v-else>Home not found</div>

      <DocNavButtons
        :prev="surroundData?.[0]"
        :next="surroundData?.[1]"
      />
    </div>
  </main>
</template>
