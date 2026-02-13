<script setup lang="ts">
import type { DocLink, DocFolder, DocItem } from "~/types/docs";
import { ACCORDION_GROUPS } from "~/stores/docs.store";

const docsStore = useDocsStore();
const { docLinks, pathToGroupMap, openGroups } = storeToRefs(docsStore);
const { isGroupOpen, toggleGroup, addOpenGroup } = docsStore;
const route = useRoute();

watch(
  () => route.path,
  (path) => {
    const groupName = pathToGroupMap.value.get(path)?.toLowerCase();
    if (groupName && ACCORDION_GROUPS.has(groupName)) {
      addOpenGroup(groupName);
    }
  },
  { immediate: true },
);

const isFolder = (item: DocLink): item is DocFolder => {
  return "items" in item;
};

const isAccordionGroup = (groupName: string) => {
  return ACCORDION_GROUPS.has(groupName.toLowerCase());
};
</script>

<template>
  <aside
    class="overflow-y-auto rounded-xl m-4 mr-0 border border-stone-200 bg-stone-50/80 shadow-md"
  >
    <div class="flex flex-col">
      <template v-for="(item, index) in docLinks" :key="index">
        <template v-if="isFolder(item)">
          <!-- 아코디언 그룹 (Alphazero, Minimax) -->
          <div
            v-if="isAccordionGroup(item.label)"
            class="border-b border-stone-200"
          >
            <button
              @click="toggleGroup(item.label)"
              class="group-btn flex w-full items-center justify-between px-4 py-3 text-left font-bold text-stone-800 hover:bg-stone-200/80 hover:text-stone-900"
            >
              <span>{{ item.label }}</span>
              <span
                :class="`pi transition-transform ${
                  isGroupOpen(item.label) ? 'pi-chevron-up' : 'pi-chevron-down'
                }`"
              />
            </button>
            <Transition name="slide-down">
              <div
                v-if="isGroupOpen(item.label)"
              >
                <NuxtLink
                  v-for="(subItem, subIndex) in item.items"
                  :key="subIndex"
                  v-ripple
                  class="sidebar-link flex items-center gap-2 pl-4 pr-2 py-2 text-stone-600 hover:bg-stone-200/70 hover:text-stone-900"
                  :to="subItem.url"
                >
                  <span :class="`pi ${subItem.icon}`" />
                  <span>{{ subItem.label }}</span>
                </NuxtLink>
              </div>
            </Transition>
          </div>

          <!-- 일반 그룹 (아코디언 없음) -->
          <div v-else class="border-b border-stone-200">
            <div class="px-4 py-3 font-bold text-stone-800">
              {{ item.label }}
            </div>
            <div>
              <NuxtLink
                v-for="(subItem, subIndex) in item.items"
                :key="subIndex"
                v-ripple
                class="sidebar-link flex items-center gap-2 pl-4 pr-2 py-2 text-stone-600 hover:bg-stone-200/70 hover:text-stone-900"
                :to="subItem.url"
              >
                <span :class="`pi ${subItem.icon}`" />
                <span>{{ subItem.label }}</span>
              </NuxtLink>
            </div>
          </div>
        </template>
      </template>
    </div>
  </aside>
</template>
<style scoped>
.sidebar-link.router-link-active {
  @apply border-l-2 border-stone-700 bg-stone-200/60 font-medium text-stone-900;
}

.sidebar-link {
  @apply border-l-2 border-transparent;
}

.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}

.slide-down-enter-from,
.slide-down-leave-to {
  max-height: 0;
  opacity: 0;
}

.slide-down-enter-to,
.slide-down-leave-from {
  max-height: 500px;
  opacity: 1;
}
</style>
