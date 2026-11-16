<script setup lang="ts">
import type { DocLink, DocItem } from "~/types/docs";

const route = useRoute();
const { docLinks } = storeToRefs(useDocsStore());

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

/** docLinks 순서(About Gomoku → About Project → Minimax → AlphaZero → Optimization)로 이전/다음 계산 */
function flattenDocLinks(links: DocLink[]): DocItem[] {
  const out: DocItem[] = [];
  for (const link of links) {
    if ("items" in link) out.push(...link.items);
    else out.push(link);
  }
  return out;
}

function normalizePathForCompare(p: string): string {
  return p.replace(/\/$/, "").toLowerCase();
}

const navSurround = computed(() => {
  const flat = flattenDocLinks(docLinks.value);
  const current = normalizePathForCompare(route.path);
  const idx = flat.findIndex((item) => normalizePathForCompare(item.url) === current);
  if (idx === -1) return { prev: null as { path: string; title: string } | null, next: null as { path: string; title: string } | null };
  return {
    prev: idx > 0 ? { path: flat[idx - 1].url, title: flat[idx - 1].label } : null,
    next: idx < flat.length - 1 ? { path: flat[idx + 1].url, title: flat[idx + 1].label } : null,
  };
});

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
      class="sticky -sm:hidden top-0 flex-shrink-0 self-start w-[280px] min-h-0 max-h-[calc(100vh-60px-92px)] mr-0"
    />

    <div class="min-h-[calc(100vh-60px-92px)] min-w-0 flex-1 border-l border-stone-200 sm:pl-6">
      <div v-if="pageData" class="min-w-0 p-6 docs-content">
        <ContentRenderer :value="pageData" />
      </div>
      <DocNotFound v-else />
      
      <DocNavButtons
        :prev="navSurround.prev"
        :next="navSurround.next"
      />
    </div>
  </main>
</template>
