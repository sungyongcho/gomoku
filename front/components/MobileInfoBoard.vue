<script setup lang="ts">
import { onClickOutside } from "@vueuse/core";

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
          class="infoEl shadow-xlg absolute right-0 top-0 z-[999] h-screen w-[300px] border-l border-black bg-white p-2 pt-8"
        >
          <button class="absolute right-0 top-0 px-4 py-3">
            <i
              class="pi pi-times text-lg text-black"
              @click="isOpen = false"
            ></i>
          </button>

          <InfoBoard />
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
