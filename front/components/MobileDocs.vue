<script setup lang="ts">
import { onClickOutside } from "@vueuse/core";

const { docLinks } = storeToRefs(useDocsStore());
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
        class="fixed inset-0 z-[998] h-screen w-full bg-black bg-opacity-50"
      >
        <div
          ref="infoEl"
          class="infoEl shadow-xlg absolute right-0 top-0 z-[999] h-screen w-[300px] border-l border-gray-500 bg-white"
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
                <span :class="`pi ${item.icon}`" />
                <span>{{ item.label }}</span>
              </NuxtLink>
            </template>
          </Menu>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped lang="scss">
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
</style>
