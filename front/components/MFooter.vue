<script setup lang="ts">
const props = defineProps({
  nonGamePage: {
    type: Boolean,
    default: false,
  },
});
</script>
<template>
  <footer class="bg-black px-8 text-white -sm:text-sm">
    <ul
      class="mx-auto grid max-w-[800px] grid-cols-2 gap-x-2 gap-y-0 p-4"
      :class="{
        ['border-t-[1px] border-white']: props.nonGamePage,
      }"
    >
      <li class="w-full text-right font-extrabold">AI & Backend by</li>
      <li class="flex flex-col">
        <a
          class="text-bold text-gray-400 underline transition-colors duration-200 hover:text-white"
          href="https://sungyongcho.com"
          target="_blank"
        >
          Sungyong Cho
        </a>
      </li>
      <li class="flex w-full items-center justify-end">
        <i class="pi pi-envelope text-sm"></i>
      </li>
      <li>
        <a
          href="mailto:here@sungyongcho.com"
          class="inline-block -translate-y-[2px] text-sm text-gray-400 underline transition-colors duration-200 hover:text-white"
          target="_blank"
          >here@sungyongcho.com</a
        >
      </li>
      <li class="w-full text-right font-extrabold">Frontend by</li>
      <li>
        <a
          class="text-bold text-gray-400 underline transition-colors duration-200 hover:text-white"
          href="https://woolimi.github.io"
          target="_blank"
          >Woolim Park</a
        >
      </li>
    </ul>
  </footer>
</template>
