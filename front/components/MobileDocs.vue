<script setup lang="ts">
import { onClickOutside } from "@vueuse/core";
import type { DocLink, DocFolder, DocItem } from "~/types/docs";
import { ACCORDION_GROUPS } from "~/stores/docs.store";

const docsStore = useDocsStore();
const { docLinks, pathToGroupMap, openGroups } = storeToRefs(docsStore);
const { isGroupOpen, toggleGroup, addOpenGroup } = docsStore;
const route = useRoute();
const isOpen = ref(false);
const infoEl = ref<HTMLElement>();

onClickOutside(infoEl, () => {
  isOpen.value = false;
});
watch(
  isOpen,
  (_isOpen) => {
    window.document.body.style.overflow = _isOpen ? "hidden" : "auto";
  },
  { immediate: true },
);

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
  <div class="relative">
    <Button
      icon="pi pi-bars"
      variant="text"
      rounded
      class="!text-white hover:!text-black lg:hidden"
      @click="isOpen = true"
    />

    <Transition name="slide-left">
      <div
        v-if="isOpen"
        class="fixed inset-0 z-[998] h-screen w-full overflow-y-auto bg-black bg-opacity-50"
      >
        <div
          ref="infoEl"
          class="infoEl shadow-xlg absolute right-0 top-0 z-[999] h-screen w-[300px] overflow-y-auto border-l border-stone-200 bg-white"
        >
          <div class="flex h-[60px] items-center justify-between bg-black px-2">
            <span class="pl-2 font-bold uppercase text-white"
              >Documentation</span
            >
            <Button
              icon="pi pi-times"
              class="!text-white"
              @click="isOpen = false"
            />
          </div>
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
                    class="flex w-full items-center justify-between px-4 py-3 text-left font-bold text-stone-800 hover:bg-stone-200/80 hover:text-stone-900"
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
                        @click="isOpen = false"
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
                      @click="isOpen = false"
                    >
                      <span :class="`pi ${subItem.icon}`" />
                      <span>{{ subItem.label }}</span>
                    </NuxtLink>
                  </div>
                </div>
              </template>
            </template>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped lang="scss">
.sidebar-link.router-link-active {
  @apply border-l-2 border-stone-700 bg-stone-200/60 font-medium text-stone-900;
}

.sidebar-link {
  @apply border-l-2 border-transparent;
}

.slide-left-enter-active {
  transition: all 0.1s ease;

  .infoEl {
    transition: all 0.2s ease;
  }
}
.slide-left-leave-active {
  transition: all 0.3s ease;
  .infoEl {
    transition: all 0.2s ease;
  }
}
.slide-left-enter-from,
.slide-left-leave-to {
  .infoEl {
    transform: translateX(100%);
  }
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
