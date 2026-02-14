<template>
  <div class="h-screen">
    <MHeader class="fixed left-0 top-0 z-10 w-full" />
    <ClientOnly>
      <slot />
      <template #fallback>
        <div
          class="flex h-screen w-full items-center justify-center bg-black"
          aria-busy="true"
        >
          <span class="text-white">Loading...</span>
        </div>
      </template>
    </ClientOnly>
    <MFooter class="fixed bottom-0 left-0 z-10 w-full" />
  </div>
</template>
