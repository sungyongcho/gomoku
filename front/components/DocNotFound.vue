<template>
  <div
    class="flex min-h-[500px] flex-col items-center justify-center gap-4 bg-white p-8 text-center"
  >
    <i class="pi pi-file-excel text-5xl" />
    <div class="space-y-1">
      <h2 class="text-xl font-bold text-black">Page not found</h2>
      <p class="text-black/70">
        The document for this path could not be found.
      </p>
    </div>
    <slot name="actions" />
  </div>
</template>
